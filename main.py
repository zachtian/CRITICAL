import gymnasium as gym
from stable_baselines3 import DQN
from stable_baselines3.common.vec_env import DummyVecEnv, VecVideoRecorder
from stable_baselines3.common.callbacks import BaseCallback
import os
import csv
import json
from collections import defaultdict
from langchain.chat_models import ChatOllama
from langchain.schema import AIMessage, HumanMessage, SystemMessage
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate

llm = ChatOllama(
    model="llama2:70b-chat",
)


env_json_schema = {
    "title": "Environment Configuration",
    "description": "Configuration settings of the simulation environment.",
    "type": "object",
    "properties": {
        # Define properties based on your environment's configuration
    },
    "required": ["lanes_count", "vehicles_count", "aggressive_vehicle_ratio"]
}

if not os.path.exists("videos"):
    os.makedirs("videos")

REAL_TIME_RENDERING = False

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
            verbose=1)

class FailureAnalysisCallback(BaseCallback):
    def __init__(self, env, verbose=0):
        super(FailureAnalysisCallback, self).__init__(verbose)
        self.failures = []
        self.env = env 
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
        if self.step_counter % 100 == 0:
            self.append_failure_stats_to_csv()
            dumps = json.dumps(env_json_schema, indent=2)
            recent_crash_types = [failure['crash_type'] for failure in self.failures]
            messages = [
                HumanMessage(content="Please analyze the following environment configuration and failure data using the JSON schema:"),
                HumanMessage(content=f"{dumps}"),
                HumanMessage(content=f"Current Environment Config: {self.env.unwrapped.config}\nRecent Failures: {recent_crash_types}"),
                HumanMessage(content="Based on this information, suggest changes to the environment configuration.")
            ]
            import pdb; pdb.set_trace()
            # Create the prompt and chain
            prompt = ChatPromptTemplate.from_messages(messages)
            chain = prompt | llm | StrOutputParser()

            # Invoke the chain to get a response
            response = chain.invoke({"dumps": dumps})
            print('LLM Suggestion:', response)
            self.failures.clear()

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


    def update_environment_config(self):
        new_config = {"aggressive_vehicle_ratio": 0.4}
        self.env.unwrapped.update_env_config(new_config)


callback = FailureAnalysisCallback(env)

model.learn(int(5e4), callback=callback)

video_env.close()