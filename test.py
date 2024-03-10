import os
import gym
from stable_baselines3 import DQN, PPO
from stable_baselines3.common.vec_env import DummyVecEnv, VecVideoRecorder
from main import FailureAnalysisCallback

if __name__ == "__main__":
    # Set the path to the trained model and the environment
    experiment_path = 'experiments/exp_PPO_False_False_1'
    model_path = os.path.join(experiment_path, 'trained_model')
    env_name = 'llm-v0'

    model_type = experiment_path.split('_')[1]

    # Load the trained model based on the model type
    if model_type == 'DQN':
        model = DQN.load(model_path)
    elif model_type == 'PPO':
        model = PPO.load(model_path)

    # Create the environment for testing
    test_env = gym.make(env_name)

    # Number of episodes for testing
    num_test_episodes = 10

    # Testing loop
    for episode in range(num_test_episodes):
        obs = test_env.reset()
        done = False
        episode_rewards = 0

        while not done:
            action, _states = model.predict(obs)
            obs, reward, done, info = test_env.step(action)
            episode_rewards += reward

            if REAL_TIME_RENDERING:
                test_env.render()

        print(f"Episode {episode + 1}: Total Reward = {episode_rewards}")

    # Close the environment
    test_env.close()
