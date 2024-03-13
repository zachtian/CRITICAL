import os
import json
import numpy as np
import pandas as pd
import gymnasium as gym
from stable_baselines3 import DQN, PPO
from stable_baselines3.common.vec_env import DummyVecEnv, VecVideoRecorder
from main import FailureAnalysisCallback
def generate_highwayenv_config(index = 0):
    df = pd.read_csv('HIGHD_data_test.csv')
    selected_row = df.iloc[index]

    aggressive_vehicle_counts = selected_row['num_aggressive']
    defensive_vehicle_counts = selected_row['num_defensive']
    regular_vehicle_counts = selected_row['num_regular']
    total_vehicles = aggressive_vehicle_counts + defensive_vehicle_counts + regular_vehicle_counts

    # Calculate ratios
    aggressive_vehicle_ratio = float(aggressive_vehicle_counts / total_vehicles)
    defensive_vehicle_ratio = float(defensive_vehicle_counts / total_vehicles)
    truck_vehicle_ratio = float(selected_row['num_trucks'] / total_vehicles)
    config = {
        "vehicles_density": selected_row['density'],
        "aggressive_vehicle_ratio": aggressive_vehicle_ratio,
        "defensive_vehicle_ratio": defensive_vehicle_ratio,
        "truck_vehicle_ratio": truck_vehicle_ratio,
        "vehicle_i_info": selected_row['vehicle_i_info'],
        "vehicle_j_info": selected_row['vehicle_j_info'],
    }

    config_json = json.dumps(config, indent=4)
    return config_json
if __name__ == "__main__":
    # Set the path to the trained model and the environment
    experiment_path = 'experiments/exp_PPO_False_True_1'
    model_path = os.path.join(experiment_path, 'trained_model')

    model_type = experiment_path.split('_')[1]

    # Load the trained model based on the model type
    if model_type == 'DQN':
        model = DQN.load(model_path)
    elif model_type == 'PPO':
        model = PPO.load(model_path)

    # Create the environment for testing
    RENDER = False
    if RENDER:
        test_env = gym.make("llm-v0", render_mode="human")
    else:
        test_env = gym.make("llm-v0")

    # Number of episodes for testing
    num_test_episodes = 10
    num_run_per_episodes =10
    # Testing loop
    for episode in range(num_test_episodes):
        rewards = []

        while len(rewards) < num_run_per_episodes:
            new_config = generate_highwayenv_config(episode)
            (obs, info), done = test_env.reset(), False
            parsed_config = json.loads(new_config[new_config.find('{'):new_config.rfind('}') + 1])
            test_env.unwrapped.update_env_config(parsed_config)

            episode_rewards = 0
            while not done:
                action, _states = model.predict(obs, deterministic=True)
                obs, reward, done, truncated, info = test_env.step(int(action))
                episode_rewards += reward
                if RENDER: 
                    test_env.render()

            if episode_rewards >= 1:
                rewards.append(episode_rewards)


        print(f"Episode {episode + 1}: Total Reward = {np.mean(rewards)}")

    # Close the environment
    test_env.close()
