import pandas as pd
import matplotlib.pyplot as plt

# Load the data
ppo_9 = pd.read_csv('/home/zach/Downloads/PPO_9.csv')
ppo_18 = pd.read_csv('/home/zach/Downloads/PPO_18.csv')
ppo_16 = pd.read_csv('/home/zach/Downloads/PPO_16.csv')

# Define the window size for smoothing
window_size = 50  # Adjust this based on your data

# Smooth the data
ppo_9_smoothed = ppo_9['Value'].rolling(window=window_size).mean()
ppo_18_smoothed = ppo_18['Value'].rolling(window=window_size).mean()
ppo_16_smoothed = ppo_16['Value'].rolling(window=window_size).mean()

# Plot the smoothed data
plt.figure(figsize=(6, 3))
plt.plot(ppo_9['Step'], ppo_9_smoothed, label='Baseline Model')
plt.plot(ppo_16['Step'], ppo_16_smoothed, label='Critical Case Generation')

plt.plot(ppo_18['Step'], ppo_18_smoothed, label='LLM-Enhanced Critical Case Generations')
plt.ylabel('Training Loss')
plt.legend()
plt.show()