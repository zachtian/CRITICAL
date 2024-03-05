import gymnasium as gym
from stable_baselines3 import DQN, PPO
from stable_baselines3.common.vec_env import DummyVecEnv, VecVideoRecorder
from stable_baselines3.common.callbacks import BaseCallback
from stable_baselines3.common.save_util import save_to_pkl
import wandb
from wandb.integration.sb3 import WandbCallback

import os
import csv
import json
import ast
import numpy as np
import pandas as pd
from collections import defaultdict, Counter
from langchain_community.chat_models import ChatOllama
from langchain.schema import AIMessage, HumanMessage, SystemMessage
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from trasnformer_utils import CustomExtractor, attention_network_kwargs
from edge_case_distribution import EdgeCaseAnalyzerFromJSON

recent_crash_types = ['front', 'front-edge', 'side-on', "rear", "rear-edge"]

def is_within_range(config, schema):
    for key in schema['properties']:
        if key in config:
            if 'minimum' in schema['properties'][key] and config[key] < schema['properties'][key]['minimum']:
                return False
            if 'maximum' in schema['properties'][key] and config[key] > schema['properties'][key]['maximum']:
                return False
    return True

class FailureAnalysisCallback(BaseCallback):
    def __init__(self, env, experiment_path, USE_LLM = False, verbose=0):
        super(FailureAnalysisCallback, self).__init__(verbose)
        self.failures = []
        self.env = env 
        self.last_obs = None
        self.step_counter = 0  
        self.failure_file = os.path.join(experiment_path, 'failures.csv')
        self.config_file = os.path.join(experiment_path, 'config.csv')
        self.HIGHD_df = 'HIGHD_data_train.csv'
        HIGHD_config = generate_highwayenv_config(self.HIGHD_df)
        self.update_environment_config(HIGHD_config)
        self.use_llm = USE_LLM
        self.episode_rewards = []
        self.episode_lengths = []
        self.cumulative_reward = 0
        self.episode_length = 0
        self.config_history = []
        self.reward_history = []
        self.edge_case_lon_and_lat_count = 0
        self.edge_case_TTC_near_miss_count = 0
        self.loop_count = 0
        self.corner_case_configurations_lat_lon = ""
        self.corner_case_configurations_ttc_near_miss = ""
        self.env_json_schema = {
            "title": "Environment Configuration",
            "description": "Configuration settings of the simulation environment.",
            "type": "object",
            "properties": {
                "vehicles_density": {
                    "type": "number",
                    "minimum": 0.2,  
                    "maximum": 10  
                },
                "aggressive_vehicle_ratio": {
                    "type": "number",
                    "minimum": 0.0, 
                    "maximum": 1.0  
                },
                "defensive_vehicle_ratio": {
                    "type": "number",
                    "minimum": 0.0,  
                    "maximum": 1.0   
                },
                "truck_vehicle_ratio": {
                    "type": "number",
                    "minimum": 0.0,  
                    "maximum": 1.0  
                }
            },
            "required": [
                "vehicles_density",
                "aggressive_vehicle_ratio", 
                "defensive_vehicle_ratio", 
                "truck_vehicle_ratio"
            ]
        }


    def _on_step(self) -> bool:
        self.step_counter += 1
        infos = self.locals['infos']
        dones = self.locals['dones']
        new_obs = self.locals['new_obs']
        actions = self.locals['actions']
        rewards = self.locals['rewards']
        self.cumulative_reward += rewards[0]
        self.episode_length += 1
        crash_type = None
        if infos[0]['crashed']:
            crash_type = infos[0].get('crash_type') 

        if self.last_obs is None:
            self.last_obs = new_obs

        near_miss_occurred, nearest_vehicle = self.env.envs[0].unwrapped.calculate_TTC_near_miss()
        if nearest_vehicle:
            is_edge_case_lon_and_lat, is_edge_case_TTC_near_miss = self.env.envs[0].unwrapped.detect_edge_case(nearest_vehicle[0], near_miss_occurred)
        else:
            is_edge_case_lon_and_lat = 0
            is_edge_case_TTC_near_miss = 0

        self.edge_case_lon_and_lat_count += is_edge_case_lon_and_lat
        self.edge_case_TTC_near_miss_count += is_edge_case_TTC_near_miss

        if dones[0]:  
            failure_info = infos[0]
            failure_type = self.determine_failure_type(failure_info)
            self.episode_rewards.append(self.cumulative_reward)
            self.episode_lengths.append(self.episode_length)
            self.cumulative_reward = 0
            self.episode_length = 0

            if failure_type is not None:
                self.failures.append({
                    "failure_type": failure_type,
                    "crash_type": crash_type,
                    "last_obs": self.last_obs,
                    "action": actions[0],
                    "reward": rewards[0],
                    "new_obs": new_obs,
                    "info": failure_info
                })

        self.last_obs = new_obs
        print("Step Number", self.step_counter)
        if self.step_counter % 100 == 0:
            self.loop_count += 1
            if self.use_llm:
                self.write_failure_stats_to_csv()
                dumps = json.dumps(self.env_json_schema, indent=2)
                recent_crash_types = [failure['crash_type'] for failure in self.failures]
                
                small_config = self.get_small_config()
                HIGHD_config = generate_highwayenv_config(self.HIGHD_df)

                self.config_history.append(small_config)
                self.reward_history.append({
                    "episode_rewards": round(np.mean(self.episode_rewards), 2),
                    "episode_lengths": np.mean(self.episode_lengths)
                })
                self.episode_rewards = []
                self.episode_lengths = []

                messages = [
                    HumanMessage(content="Based on the following data and constraints, suggest modifications for scenario generation. Provide the updated configuration as a JSON dictionary within the specified ranges."),
                    HumanMessage(content=f"JSON Schema: {dumps}"),
                    HumanMessage(content=f"Real-World Traffic Data: {HIGHD_config}"),
                    HumanMessage(content=f"Current Config: {small_config}"),
                    HumanMessage(content=f"Recent Failures: {Counter(recent_crash_types)}"),
                    HumanMessage(content=f"Longitude and Latitude-based Edge Cases: {float(self.edge_case_lon_and_lat_count)}"),
                    HumanMessage(content=f"Time to Collision (TTC) Near Miss Cases: {float(self.edge_case_TTC_near_miss_count)}"),
                ]

                for i in range(1, min(len(self.config_history) - 1, 5)  + 1):
                    step_number = len(self.config_history) - i 
                    config_index = -i - 1  
                    messages.append(HumanMessage(content=f"Previous Episode Config {step_number}: {self.config_history[config_index]}"))
                    messages.append(HumanMessage(content=f"Previous Episode Lengths {step_number}: {self.reward_history[config_index]}"))

                messages.append(
                    HumanMessage(content="Considering the insights from the Real-World Traffic Data and the observed simulation trends, suggest modifications to enhance scenario realism. Focus on adjusting the following properties of the environment configuration to reflect real-world driving behaviors and patterns: vehicles_density, aggressive_vehicle_ratio, defensive_vehicle_ratio, truck_vehicle_ratio. Your response should be a JSON dictionary containing only these four elements.")
                )
                prompt = ChatPromptTemplate.from_messages(messages)
                chain = prompt | llm | StrOutputParser()

                self.failures.clear()

                for attempt in range(5):
                    try:
                        response = chain.invoke({"dumps": dumps})
                        parsed_config = json.loads(response[response.find('{'):response.rfind('}') + 1])
                        if is_within_range(parsed_config, json.loads(dumps)):
                            print('LLM Suggestion:', response)
                            self.update_environment_config(response)
                            break
                        else:
                            messages.append(
                                HumanMessage(content="PLEASE DOUBLE CHECK IF THE SUGGESTED VALUES ARE WHITIN RANGE.")
                            )
                            prompt = ChatPromptTemplate.from_messages(messages)
                            chain = prompt | llm | StrOutputParser()

                    except Exception as e:
                        print(response)
                        print("Error updating environment config:", e)
                        if attempt == 4:
                            import pdb; pdb.set_trace()

            else:
                HIGHD_config = generate_highwayenv_config(self.HIGHD_df)
                self.update_environment_config(HIGHD_config)

            edge_case_values_dict = {
                "edge_case_count_for_lat_and_lon": float(self.edge_case_lon_and_lat_count),
                "edge_case_count_for_TTC_near_miss": float(self.edge_case_TTC_near_miss_count)              
            }
            self.write_config_to_json(edge_case_values_dict, self.config_file, optional_index= (self.loop_count - 1))
            self.edge_case_lon_and_lat_count = 0
            self.edge_case_TTC_near_miss_count = 0

        if (self.step_counter != 0) and (self.step_counter % 500 == 0):
            configuration_file_name = f"corner_case_configurations/exp_{RL_MODEL}_{USE_LLM}_{next_exp_number}/for_step_{self.step_counter}"

            directory = os.path.dirname(configuration_file_name)
            if not os.path.exists(directory):
                os.makedirs(directory)
            
            file_path = f'experiments/exp_{RL_MODEL}_{USE_LLM}_{next_exp_number}/config.csv'
            analyzer = EdgeCaseAnalyzerFromJSON(file_path)
            analyzer.plot_distribution()  # Must be called before the other methods
            lat_lon_configs_str, ttc_near_miss_configs_str = analyzer.get_configurations_for_last_bins()

            self.corner_case_configurations_lat_lon = "Configurations for last bins of Lat & Lon Edge Cases:\n" + "\n".join(lat_lon_configs_str)
            self.corner_case_configurations_ttc_near_miss = "Configurations for last bins of TTC Near Miss Cases:\n" + "\n".join(ttc_near_miss_configs_str)

            self.write_config_to_json(self.corner_case_configurations_lat_lon, configuration_file_name)
            self.write_config_to_json(self.corner_case_configurations_ttc_near_miss, configuration_file_name)

        if REAL_TIME_RENDERING:
            self.model.env.render()

        return True
            
    def determine_failure_type(self, info):
        if 'crashed' in info and info['crashed']:
            return 'crashed'
        if 'went_offroad' in info and info['went_offroad']:
            return 'went_offroad'

        return None

    def get_small_config(self):
        inner_env = self.env.envs[0]
        full_config = dict(inner_env.unwrapped.config)
        small_config = {k: full_config[k] for k in ('vehicles_density', 'aggressive_vehicle_ratio', 'defensive_vehicle_ratio', 'truck_vehicle_ratio')}

        return small_config

    def update_environment_config(self, new_config):
        parsed_config = json.loads(new_config[new_config.find('{'):new_config.rfind('}') + 1])
        inner_env = self.env.envs[0]
        inner_env.unwrapped.update_env_config(parsed_config)

        self.write_config_to_json(parsed_config, self.config_file)

    def write_failure_stats_to_csv(self):
        with open(self.failure_file, 'a', newline='') as file:
            writer = csv.writer(file)
            
            # Write header if the file is new
            if file.tell() == 0:
                writer.writerow(['Step', 'Failure Type', 'Count', 'Crash Type', 'Crash Count'])

            # Aggregate failure statistics
            failure_counts = defaultdict(int)
            crash_type_counts = defaultdict(lambda: defaultdict(int))
            for failure in self.failures:
                failure_type = failure["failure_type"]
                crash_type = failure["crash_type"]
                failure_counts[failure_type] += 1
                crash_type_counts[failure_type][crash_type] += 1

            # Write failure statistics to CSV
            for failure_type, f_count in failure_counts.items():
                for crash_type, c_count in crash_type_counts[failure_type].items():
                    writer.writerow([self.step_counter, failure_type, f_count, crash_type, c_count])

    def write_config_to_json(self, config, file_name, optional_index=None):
        # Load existing data or start with an empty list if file doesn't exist
        try:
            with open(file_name, 'r') as json_file:
                existing_data = json.load(json_file)
                if not isinstance(existing_data, list):
                    existing_data = [existing_data]
        except (FileNotFoundError, json.JSONDecodeError):
            existing_data = []

        if optional_index == None:
            # Append new configuration
            existing_data.append(config)

            # Write updated data back to JSON file
            with open(file_name, 'w') as json_file:
                json.dump(existing_data, json_file, indent=4)
        else: 
            # Add new configuration
            existing_data[optional_index].update(config)

            # Write updated data back to JSON file
            with open(file_name, 'w') as json_file:
                json.dump(existing_data, json_file, indent=4)

def generate_highwayenv_config(csv_file):
    df = pd.read_csv(csv_file)
    selected_row = df.sample(n=1).iloc[0]

    aggressive_vehicle_counts = selected_row['num_aggressive']
    defensive_vehicle_counts = selected_row['num_defensive']
    regular_vehicle_counts = selected_row['num_regular']
    total_vehicles = aggressive_vehicle_counts + defensive_vehicle_counts + regular_vehicle_counts

    # Calculate ratios
    aggressive_vehicle_ratio = float(aggressive_vehicle_counts / total_vehicles)
    defensive_vehicle_ratio = float(defensive_vehicle_counts / total_vehicles)
    truck_vehicle_ratio = float(selected_row['num_trucks'] / total_vehicles)
    config = {
        "vehicles_density": selected_row['density'] /2.5,
        "aggressive_vehicle_ratio": aggressive_vehicle_ratio,
        "defensive_vehicle_ratio": defensive_vehicle_ratio,
        "truck_vehicle_ratio": truck_vehicle_ratio,
        "vehicle_i_info": selected_row['vehicle_i_info'],
        "vehicle_j_info": selected_row['vehicle_j_info'],
    }

    config_json = json.dumps(config, indent=4)
    return config_json

llm = ChatOllama(
    model="llama2:13b",
)


if __name__ == "__main__":
    if not os.path.exists("videos"):
        os.makedirs("videos")

    REAL_TIME_RENDERING = True
    USE_LLM = False
    POLICY_NET = 'mlp'
    RL_MODEL = 'PPO'
    if not os.path.exists('experiments'):
        os.makedirs('experiments', exist_ok=True)

    next_exp_number = 1
    while os.path.exists(os.path.join('experiments', f"exp_{RL_MODEL}_{USE_LLM}_{next_exp_number}")):
        next_exp_number += 1

    experiment_path = os.path.join('experiments', f"exp_{RL_MODEL}_{USE_LLM}_{next_exp_number}")
    os.makedirs(experiment_path, exist_ok=True)

    env = DummyVecEnv([lambda: gym.make('llm-v0', render_mode='rgb_array')])
    if REAL_TIME_RENDERING:
        env = VecVideoRecorder(env, "videos", 
                                    record_video_trigger=lambda step: step % 100 == 0,
                                    video_length=50, 
                                    name_prefix=f"{RL_MODEL}_highway")

    if POLICY_NET == 'transformer':
        policy_kwargs = dict(
                features_extractor_class=CustomExtractor,
                features_extractor_kwargs=attention_network_kwargs,
        )
    elif POLICY_NET == 'mlp':
        if RL_MODEL == 'DQN':
            policy_kwargs=dict(net_arch=[256, 256])
        else:
            policy_kwargs = {
                "net_arch": dict(pi=[256, 256], vf=[256, 256]) 
                }

    if RL_MODEL == 'DQN':
        model = DQN(
            "MlpPolicy", 
            env,
            policy_kwargs=policy_kwargs,
            learning_rate=1e-3,
            buffer_size=30000,
            learning_starts=200,
            batch_size=256,
            gamma=0.8,
            train_freq=1,
            gradient_steps=1,
            target_update_interval=50,
            exploration_fraction=0.5,
            verbose=1,
            tensorboard_log='logs',
        )
    else:
        model = PPO(
            "MlpPolicy",
            env,
            policy_kwargs=policy_kwargs,
            n_steps=256,
            batch_size=256,
            n_epochs=10,
            learning_rate=5e-4,
            gamma=0.8,
            verbose=2,
            tensorboard_log='logs',
        )
    # wandb.init(project="LLMAV", name=f"exp_kethan_{RL_MODEL}_{USE_LLM}_{next_exp_number}",sync_tensorboard=True)
    # wandb_callback = WandbCallback(
    #     gradient_save_freq=1000,  # adjust according to your needs
    #     model_save_path=f"{wandb.run.dir}/model",  # save model in wandb directory
    #     verbose=2,
    # )
    callback = FailureAnalysisCallback(env, experiment_path, USE_LLM)
    model.learn(int(2e5), callback=[callback])
    model.save(os.path.join(experiment_path, 'trained_model'))

    env.close()