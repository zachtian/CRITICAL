import os
import gymnasium as gym
from stable_baselines3 import DQN, PPO
from stable_baselines3.common.vec_env import DummyVecEnv, VecVideoRecorder
from main import FailureAnalysisCallback

if __name__ == "__main__":
    # Set the path to the trained model and the environment
    experiment_path = 'experiments/exp_PPO_False_False_1'
    model_path = os.path.join(experiment_path, 'trained_model')

    model_type = experiment_path.split('_')[1]

    # Load the trained model based on the model type
    if model_type == 'DQN':
        model = DQN.load(model_path)
    elif model_type == 'PPO':
        model = PPO.load(model_path)

    # Create the environment for testing
    test_env = gym.make("llm-v0", render_mode="rgb_array")

    # Number of episodes for testing
    num_test_episodes = 10

    # Testing loop
    for episode in range(num_test_episodes):
        (obs, info), done = test_env.reset(), False
        done = False
        episode_rewards = 0

        while not done:
            action, _states = model.predict(obs, deterministic=True)
            obs, reward, done, truncated, info = test_env.step(int(action))
            episode_rewards += reward
            test_env.render()

        print(f"Episode {episode + 1}: Total Reward = {episode_rewards}")

    # Close the environment
    test_env.close()
