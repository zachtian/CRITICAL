import gymnasium as gym
from stable_baselines3 import DQN
from stable_baselines3.common.vec_env import DummyVecEnv, VecVideoRecorder
from stable_baselines3.common.callbacks import BaseCallback
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

recent_crash_types = ['front', 'front-edge', 'side-on', "rear", "rear-edge"]
class FailureAnalysisCallback(BaseCallback):
    def __init__(self, env, USE_LLM = False, verbose=0):
        super(FailureAnalysisCallback, self).__init__(verbose)
        self.failures = []
        self.env = env 
        self.last_obs = None
        self.step_counter = 0  
        self.csv_file = 'failures.csv'
        self.NGSIM_df = pd.read_csv('NGSIM_data.csv')
        NGSIM_config = generate_highwayenv_config(self.NGSIM_df)
        self.update_environment_config(NGSIM_config)
        self.use_llm = USE_LLM
        self.attempts_to_generate_valid_config = 100
        self.format_check = False
        self.episode_rewards = []
        self.episode_lengths = []
        self.cumulative_reward = 0
        self.episode_length = 0
        self.config_history = []
        self.reward_history = []

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
            print(f"Crash type: {crash_type}")

        if infos[0]['went_offroad']:
            print("went_offroad!")

        if self.last_obs is None:
            self.last_obs = new_obs  

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
            if self.use_llm:
                self.append_failure_stats_to_csv()
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
                        HumanMessage(content="Based on this analysis, suggest realistic modifications to the environment configuration that can improve the training of the autonomous vehicle agent. Please focus on practical and achievable changes, avoiding the introduction of new parameters.")
                    ]
                else:
                    messages = [
                        HumanMessage(content="Please analyze the following data and suggest modifications for scenario generation:"),
                        HumanMessage(content=f"JSON Schema: {dumps}"),
                        HumanMessage(content=f"Real-World Traffic Data (NGSIM_config): {NGSIM_config}"),
                        HumanMessage(content=f"Simulation Environment Config: {small_config}"),
                        HumanMessage(content=f"Recent Failures: {Counter(recent_crash_types)}"),
                        HumanMessage(content="Based on this analysis, suggest realistic modifications to the environment configuration that can improve the training of the autonomous vehicle agent. Please focus on practical and achievable changes, avoiding the introduction of new parameters.")
                    ]               
                prompt = ChatPromptTemplate.from_messages(messages)
                chain = prompt | llm | StrOutputParser()

                response = chain.invoke({"dumps": dumps})
                print('LLM Suggestion:', response)
                self.failures.clear()

                # Use LLM response to edit environment configuration file 
                new_config = self.generate_new_config_file(small_config, response)
                print(new_config)

                for i in range(self.attempts_to_generate_valid_config):
                    if self.format_check == False:
                        try:
                            self.update_environment_config(new_config)
                            self.format_check = True
                        except:
                            print("The chosen LLM did not appropriately format its environment configuration suggestions, trying again.")
                    else:
                        break
            else:
                NGSIM_config = generate_highwayenv_config(self.NGSIM_df)
                self.update_environment_config(NGSIM_config)

        if REAL_TIME_RENDERING:
            self.model.env.render()

        return True

    def write_config_to_json(self, config, file_name):
        # Check if file exists
        try:
            with open(file_name, 'r') as json_file:
                try:
                    # Try to read the existing data
                    existing_data = json.load(json_file)
                except json.JSONDecodeError:
                    # If the file is empty, start with an empty list
                    existing_data = []
        except FileNotFoundError:
            # If the file does not exist, start with an empty list
            existing_data = []

        # Ensure that existing_data is a list
        if not isinstance(existing_data, list):
            existing_data = [existing_data]

        # Append the new configuration to the list
        existing_data.append(config)

        # Write the updated list of configurations back to the file
        with open(file_name, 'w') as json_file:
            json.dump(existing_data, json_file, indent=4)
            
    def determine_failure_type(self, info):
        if 'crashed' in info and info['crashed']:
            return 'crashed'
        if 'went_offroad' in info and info['went_offroad']:
            return 'went_offroad'
        return None

    def print_failure_stats(self):
        # Print the categorized failure counts
        failure_counts = defaultdict(int)
        for failure in self.failures:
            failure_type = failure["failure_type"]
            failure_counts[failure_type] += 1

        print(f"Failure Types and Counts at step {self.step_counter}:")
        for failure_type, count in failure_counts.items():
            print(f"{failure_type}: {count}")

    def append_failure_stats_to_csv(self):
        # Categorize and count failures and crash types
        failure_counts = defaultdict(int)
        crash_type_counts = defaultdict(lambda: defaultdict(int))
        for failure in self.failures:
            failure_type = failure["failure_type"]
            failure_counts[failure_type] += 1
            crash_type = failure["crash_type"]
            crash_type_counts[failure_type][crash_type] += 1

        with open(self.csv_file, 'a', newline='') as file:
            writer = csv.writer(file)
            if file.tell() == 0:
                writer.writerow(['Step', 'Failure Type', 'Count', 'Crash Type', 'Crash Count'])
            for failure_type, f_count in failure_counts.items():
                for crash_type, c_count in crash_type_counts[failure_type].items():
                    writer.writerow([self.step_counter, failure_type, f_count, crash_type, c_count])

    def generate_new_config_file(self, small_config, response) -> dict:
        dumps = json.dumps(env_json_schema, indent=2)
        messages = [
            HumanMessage(content=f"Given the following current environment configuration: {small_config}"),
            HumanMessage(content=f"And the following editing suggestions: {response}"),
            HumanMessage(content="Please generate an updated configuration file in dictionary format. ONLY generate the new configuration file in dictionary format, no other text. For example, do not include config= or an introductory or closing sentence, or anything like that")
        ]

        prompt = ChatPromptTemplate.from_messages(messages)
        chain = prompt | llm | StrOutputParser()
        response_for_updated_config = chain.invoke({"dumps": dumps})

        return response_for_updated_config

    def get_small_config(self):
        full_config = dict(self.env.unwrapped.config)
        small_config = {k: full_config[k] for k in ('vehicles_density', 'aggressive_vehicle_ratio', 'defensive_vehicle_ratio', 'truck_vehicle_ratio', 'motor_vehicle_ratio')}

        return small_config

    def update_environment_config(self, new_config):
        parsed_config = ast.literal_eval(new_config)
        self.env.unwrapped.update_env_config(new_config)
        self.write_config_to_json(parsed_config, "updated_config.json")

def generate_highwayenv_config(df):
    selected_row = df.sample(n=1).iloc[0]

    driving_style_dict = ast.literal_eval(selected_row['Driving_Behavior'])
    vehicle_class_dict = ast.literal_eval(selected_row['Vehicle_Type'])
    aggressive_vehicle_counts = driving_style_dict.get('Aggressive', 0)
    defensive_vehicle_counts = driving_style_dict.get('Defensive', 0)
    regular_vehicle_counts = driving_style_dict.get('Regular', 0)
    total_vehicles = aggressive_vehicle_counts + defensive_vehicle_counts + regular_vehicle_counts
    # Calculate ratios
    aggressive_vehicle_ratio = float(aggressive_vehicle_counts / total_vehicles)
    defensive_vehicle_ratio = float(defensive_vehicle_counts / total_vehicles)
    truck_vehicle_ratio = float(vehicle_class_dict.get('Truck', 0) / total_vehicles)
    motor_vehicle_ratio = float(vehicle_class_dict.get('Motorcycle', 0) / total_vehicles)

    config = {
        "aggressive_vehicle_ratio": aggressive_vehicle_ratio,
        "defensive_vehicle_ratio": defensive_vehicle_ratio,
        "truck_vehicle_ratio": truck_vehicle_ratio,
        "motor_vehicle_ratio": motor_vehicle_ratio,
    }

    config_json = json.dumps(config, indent=4)

    return config_json

llm = ChatOllama(
    model="llama2:70b-chat",
)

env_json_schema = {
    "title": "Environment Configuration",
    "description": "Configuration settings of the simulation environment.",
    "type": "object",
    "properties": {
        "vehicles_density": {"type": "number"},
        "aggressive_vehicle_ratio": {"type": "number"},
        "defensive_vehicle_ratio": {"type": "number"},
        "truck_vehicle_ratio": {"type": "number"},
        "motor_vehicle_ratio": {"type": "number"}
    },
    "required": [
        "vehicles_density",
        "aggressive_vehicle_ratio", 
        "defensive_vehicle_ratio", 
        "truck_vehicle_ratio", 
        "motor_vehicle_ratio"
    ]
}

if __name__ == "__main__":
    if not os.path.exists("videos"):
        os.makedirs("videos")

    REAL_TIME_RENDERING = False
    USE_LLM = True

    if REAL_TIME_RENDERING:
        env = DummyVecEnv([lambda: gym.make('llm-v0', render_mode='rgb_array')])
        env = VecVideoRecorder(env, "videos", 
                                    record_video_trigger=lambda step: step % 100 == 0,
                                    video_length=50, 
                                    name_prefix="dqn_highway")
    else:
        env = gym.make('llm-v0')

    model = DQN('MlpPolicy', env,
                policy_kwargs=dict(net_arch=[256, 256]),
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

    callback = FailureAnalysisCallback(env, USE_LLM)

    model.learn(int(2e5), callback=callback)

    video_env.close()