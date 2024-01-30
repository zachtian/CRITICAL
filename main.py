import gymnasium as gym
import highway_env

import sys
import gymnasium as gym
import highway_env
from stable_baselines3 import DQN
import csv

from rl_agents.agents.common.factory import agent_factory

import sys
from tqdm.notebook import trange
sys.path.insert(0, '/home/zach/HighwayEnv/scripts')
from utils import record_videos, show_videos
from stable_baselines3.common.callbacks import BaseCallback
from collections import defaultdict

class FailureAnalysisCallback(BaseCallback):
    def __init__(self, verbose=0):
        super(FailureAnalysisCallback, self).__init__(verbose)
        self.failures = []
        self.last_obs = None
        self.step_counter = 0  # Initialize a step counter
        self.csv_file = 'failures.csv'

    def _on_step(self) -> bool:
        # Increment step counter
        self.step_counter += 1

        # Access local variables from the training loop
        infos = self.locals['infos']
        dones = self.locals['dones']
        new_obs = self.locals['new_obs']
        actions = self.locals['actions']
        rewards = self.locals['rewards']
        if infos[0]['crashed']:
            crash_type = infos[0].get('crash_type')  # Safely get crash_type
            print(f"Crash type: {crash_type}")

        # Store the last observation for failure analysis
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

        # Update the last observation
        self.last_obs = new_obs

        # Print failure types and counts every 5,000 steps
        if self.step_counter % 1000 == 0:
            self.append_failure_stats_to_csv()

        return True

    def determine_failure_type(self, info):
        # Determine the type of failure from info
        if 'crashed' in info and info['crashed']:
            return 'crashed'
        elif 'offroad' in info and info['offroad']:
            return 'offroad'
        elif 'timeout' in info:
            return 'timeout'
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

        # Append the categorized failure and crash type counts to CSV
        with open(self.csv_file, 'a', newline='') as file:
            writer = csv.writer(file)
            # Write header if file is empty
            if file.tell() == 0:
                writer.writerow(['Step', 'Failure Type', 'Count', 'Crash Type', 'Crash Count'])
            # Write failure and crash type counts
            for failure_type, f_count in failure_counts.items():
                for crash_type, c_count in crash_type_counts[failure_type].items():
                    writer.writerow([self.step_counter, failure_type, f_count, crash_type, c_count])

        # Clear failures list after saving to CSV
        self.failures.clear()

model = DQN('MlpPolicy', 'llm-v0',
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
                verbose=1,)

callback = FailureAnalysisCallback()
model.learn(int(5e4), callback=callback)