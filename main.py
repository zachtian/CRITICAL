import gymnasium as gym
from stable_baselines3 import DQN, PPO
from stable_baselines3.common.vec_env import DummyVecEnv, VecVideoRecorder
from stable_baselines3.common.callbacks import BaseCallback
from stable_baselines3.common.save_util import save_to_pkl

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

recent_crash_types = ['front', 'front-edge', 'side-on', "rear", "rear-edge"]

class FailureAnalysisCallback(BaseCallback):
    def __init__(self, env, experiment_path, USE_LLM = False, verbose=0):
        super(FailureAnalysisCallback, self).__init__(verbose)
        self.failures = []
        self.env = env 
        self.last_obs = None
        self.step_counter = 0  
        self.failure_file = os.path.join(experiment_path, 'failures.csv')
        self.config_file = os.path.join(experiment_path, 'config.csv')
        self.NGSIM_df = 'highwayenv_scenario_data.csv'
        NGSIM_config = generate_highwayenv_config(self.NGSIM_df)
        self.update_environment_config(NGSIM_config)
        self.use_llm = USE_LLM
        self.attempts_to_generate_valid_config = 100
        self.episode_rewards = []
        self.episode_lengths = []
        self.cumulative_reward = 0
        self.episode_length = 0
        self.config_history = []
        self.reward_history = []
        self.edge_case_lon_and_lat_count = 0
        self.edge_case_TTC_near_miss_count = 0
        self.loop_count = 0

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
        is_edge_case_lon_and_lat, is_edge_case_TTC_near_miss = self.env.envs[0].unwrapped.detect_edge_case(nearest_vehicle[0], near_miss_occurred)

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
        if self.step_counter % 500 == 0:
            self.loop_count += 1
            if self.use_llm:
                self.write_failure_stats_to_csv()
                dumps = json.dumps(env_json_schema, indent=2)
                recent_crash_types = [failure['crash_type'] for failure in self.failures]
                
                small_config = self.get_small_config()
                NGSIM_config = generate_highwayenv_config(self.NGSIM_df)

                self.config_history.append(small_config)
                self.reward_history.append({
                    "episode_rewards": round(np.mean(self.episode_rewards),2),
                    "episode_lengths": np.mean(self.episode_lengths)
                })
                self.episode_rewards = []
                self.episode_lengths = []
                if len(self.config_history) > 1:
                    messages = [
                        HumanMessage(content="Please analyze the following data and suggest modifications for scenario generation:"),
                        HumanMessage(content=f"JSON Schema: {dumps}"),
                        HumanMessage(content=f"Real-World Traffic Data (NGSIM_config): {NGSIM_config}"),
                        HumanMessage(content=f"Simulation Environment Config: {small_config}"),
                        HumanMessage(content=f"Recent Failures: {Counter(recent_crash_types)}"),
                        HumanMessage(content=f"Previous Episode Config: {self.config_history[-2]}"),
                        HumanMessage(content=f"Previous Episode Lengths: {self.reward_history[-2]}"),
                        HumanMessage(content="Based on this analysis, provide modifications only for the following properties in the environment configuration: vehicles_density, aggressive_vehicle_ratio, defensive_vehicle_ratio, truck_vehicle_ratio. Avoid adding new parameters or unrelated content.")
                    ]
                else:
                    messages = [
                        HumanMessage(content="Please analyze the following data and suggest modifications for scenario generation:"),
                        HumanMessage(content=f"JSON Schema: {dumps}"),
                        HumanMessage(content=f"Real-World Traffic Data (NGSIM_config): {NGSIM_config}"),
                        HumanMessage(content=f"Simulation Environment Config: {small_config}"),
                        HumanMessage(content=f"Recent Failures: {Counter(recent_crash_types)}"),
                        HumanMessage(content="Based on this analysis, provide modifications only for the following properties in the environment configuration: vehicles_density, aggressive_vehicle_ratio, defensive_vehicle_ratio, truck_vehicle_ratio. Avoid adding new parameters or unrelated content.")
                    ]               
                prompt = ChatPromptTemplate.from_messages(messages)
                chain = prompt | llm | StrOutputParser()

                response = chain.invoke({"dumps": dumps})
                print('LLM Suggestion:', response)
                self.failures.clear()

                edge_case_values_dict = {
                    "edge_case_count_for_lat_and_lon": float(self.edge_case_lon_and_lat_count),
                    "edge_case_count_for_TTC_near_miss": float(self.edge_case_TTC_near_miss_count)              
                }
                self.write_config_to_json(edge_case_values_dict, self.config_file, optional_index=(self.loop_count-1))

                self.edge_case_lon_and_lat_count = 0
                self.edge_case_TTC_near_miss_count = 0
                
                new_config = self.generate_new_config_file(small_config, response)
                try:
                    self.update_environment_config(new_config)
                except:
                    import pdb; pdb.set_trace()

            else:
                edge_case_values_dict = {
                    "edge_case_count_for_lat_and_lon": float(self.edge_case_lon_and_lat_count),
                    "edge_case_count_for_TTC_near_miss": float(self.edge_case_TTC_near_miss_count)              
                }
                self.write_config_to_json(edge_case_values_dict, self.config_file)
                self.edge_case_lon_and_lat_count = 0
                self.edge_case_TTC_near_miss_count = 0

                NGSIM_config = generate_highwayenv_config(self.NGSIM_df)
                self.update_environment_config(NGSIM_config)

        if REAL_TIME_RENDERING:
            self.model.env.render()

        return True
            
    def determine_failure_type(self, info):
        if 'crashed' in info and info['crashed']:
            return 'crashed'
        if 'went_offroad' in info and info['went_offroad']:
            return 'went_offroad'
        return None

    def generate_new_config_file(self, small_config, response) -> dict:
        dumps = json.dumps(env_json_schema, indent=2)
        messages = [
            HumanMessage(content="Given the current environment configuration and the following suggestions, please generate an updated configuration file."),
            HumanMessage(content=f"Current Configuration Schema: {dumps}"),
            HumanMessage(content=f"Current Configuration: {small_config}"),
            HumanMessage(content=f"Suggested Modifications: {response}"),
            HumanMessage(content="Please update the configuration based on these suggestions, adhering to the current schema. Generate only the updated configuration in dictionary format, without additional text or comments.")
        ]
        prompt = ChatPromptTemplate.from_messages(messages)
        chain = prompt | llm | StrOutputParser()
        response_for_updated_config = chain.invoke({"dumps": dumps})

        return response_for_updated_config

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
        "aggressive_vehicle_ratio": aggressive_vehicle_ratio,
        "defensive_vehicle_ratio": defensive_vehicle_ratio,
        "truck_vehicle_ratio": truck_vehicle_ratio,
    }

    config_json = json.dumps(config, indent=4)
    return config_json

llm = ChatOllama(
    model="llama2:13b-chat",
)

env_json_schema = {
    "title": "Environment Configuration",
    "description": "Configuration settings of the simulation environment.",
    "type": "object",
    "properties": {
        "vehicles_density": {"type": "number"},
        "aggressive_vehicle_ratio": {"type": "number"},
        "defensive_vehicle_ratio": {"type": "number"},
        "truck_vehicle_ratio": {"type": "number"}
    },
    "required": [
        "vehicles_density",
        "aggressive_vehicle_ratio", 
        "defensive_vehicle_ratio", 
        "truck_vehicle_ratio"
    ]
}

if __name__ == "__main__":
    if not os.path.exists("videos"):
        os.makedirs("videos")

    REAL_TIME_RENDERING = False
    USE_LLM = True
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
            #tensorboard_log='logs',
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

    callback = FailureAnalysisCallback(env, experiment_path, USE_LLM)
    model.learn(int(2e5), callback=callback)
    model.save(os.path.join(experiment_path, 'trained_model'))

    env.close()