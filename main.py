import gymnasium as gym
from stable_baselines3 import DQN
from stable_baselines3.common.vec_env import DummyVecEnv, VecVideoRecorder
from stable_baselines3.common.callbacks import BaseCallback
import os
import csv
import json
import ast

import pandas as pd
from collections import defaultdict
from langchain_community.chat_models import ChatOllama
from langchain.schema import AIMessage, HumanMessage, SystemMessage
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate


class FailureAnalysisCallback(BaseCallback):
    def __init__(self, env, USE_LLM = False, verbose=0):
        super(FailureAnalysisCallback, self).__init__(verbose)
        self.failures = []
        self.env = env 
        self.last_obs = None
        self.step_counter = 0  # Initialize a step counter
        self.csv_file = 'failures.csv'
        self.NGSIM_df = pd.read_csv('NGSIM_data.csv')
        NGSIM_config = generate_highwayenv_config(self.NGSIM_df)
        self.update_environment_config(NGSIM_config)
        self.use_llm = USE_LLM
        self.attempts_to_generate_valid_config = 100
        self.format_check = False

    def _on_step(self) -> bool:
        # Increment step counter
        self.step_counter += 1
        # Access local variables from the training loop
        infos = self.locals['infos']
        dones = self.locals['dones']
        new_obs = self.locals['new_obs']
        actions = self.locals['actions']
        rewards = self.locals['rewards']
        crash_type = None
        if infos[0]['crashed']:
            crash_type = infos[0].get('crash_type')  # Safely get crash_type
            print(f"Crash type: {crash_type}")

        if infos[0]['went_offroad']:
            print("went_offroad!")

        if self.last_obs is None:
            self.last_obs = new_obs  # This is the first observation

        # Check if the episode ended and why
        if dones[0]:  # Assuming a single environment
            failure_info = infos[0]
            failure_type = self.determine_failure_type(failure_info)

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
        if self.use_llm and self.step_counter % 500 == 0:
            self.append_failure_stats_to_csv()
            dumps = json.dumps(env_json_schema, indent=2)
            recent_crash_types = [failure['crash_type'] for failure in self.failures]
            
            small_config = self.get_small_config()
            NGSIM_config = generate_highwayenv_config(self.NGSIM_df)
            messages = [
                HumanMessage(content="Please analyze the following environment configuration and failure data using the JSON schema:"),
                HumanMessage(content=f"{dumps}"),
                HumanMessage(content="Current Real-World Traffic Data (NGSIM_config):"),
                HumanMessage(content=f"{NGSIM_config}"),
                HumanMessage(content=f"Current Simulation Environment Config: {small_config}\nRecent Failures: {recent_crash_types}"),
                HumanMessage(content="Based on the real-world data and current simulation settings, suggest changes to the environment configuration. Please consider the NGSIM_config as a reference for realism. Do not add additional parameters. Only suggest edits to existing parameters.")
            ]
            # Create the prompt and chain
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

        if REAL_TIME_RENDERING:
            self.model.env.render()

        return True

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

        # Create the prompt and chain
        prompt = ChatPromptTemplate.from_messages(messages)
        chain = prompt | llm | StrOutputParser()
        # Invoke the chain to get a response
        response_for_updated_config = chain.invoke({"dumps": dumps})

        return response_for_updated_config

    def get_small_config(self):
        full_config = dict(self.env.unwrapped.config)
        small_config = {k: full_config[k] for k in ('vehicles_count', 'vehicles_density', 'aggressive_vehicle_ratio', 'defensive_vehicle_ratio', 'truck_vehicle_ratio', 'motor_vehicle_ratio')}

        return small_config

    def update_environment_config(self, new_config):
        self.env.unwrapped.update_env_config(new_config)

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
    model="llama2:13b-chat",
)

env_json_schema = {
    "title": "Environment Configuration",
    "description": "Configuration settings of the simulation environment.",
    "type": "object",
    "properties": {
        "vehicles_count": {"type": "integer"},
        "vehicles_density": {"type": "number"},
        "aggressive_vehicle_ratio": {"type": "number"},
        "defensive_vehicle_ratio": {"type": "number"},
        "truck_vehicle_ratio": {"type": "number"},
        "motor_vehicle_ratio": {"type": "number"}
    },
    "required": [
        "vehicles_count", 
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
                learning_rate=5e-4,
                buffer_size=15000,
                learning_starts=200,
                batch_size=128,
                gamma=0.8,
                train_freq=1,
                gradient_steps=1,
                target_update_interval=50,
                exploration_fraction=0.7,
                verbose=1,
                tensorboard_log='logs',)

    callback = FailureAnalysisCallback(env, USE_LLM)

    model.learn(int(1e5), callback=callback)

    video_env.close()