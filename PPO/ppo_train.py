import numpy as np
import torch
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv
from stable_baselines3.common.evaluation import evaluate_policy
from stable_baselines3.common.env_checker import check_env
from environment import PongEnvironment
import matplotlib.pyplot as plt
import os
import gymnasium as gym


class PongEnvWrapper(gym.Env):
    def __init__(self):
        super().__init__()
        self.env = PongEnvironment()
        self.action_space = gym.spaces.Discrete(2)
        self.observation_space = gym.spaces.Box(low=-np.inf, high=np.inf, shape=(6,), dtype=np.float32)

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self.env.reset()
        return np.array(self.env.get_striker_and_ball_coordinates(), dtype=np.float32), {}

    def step(self, action):
        _, reward, done, info = self.env.step(action)
        state = np.array(self.env.get_striker_and_ball_coordinates(), dtype=np.float32)
        return state, reward, done, False, info

    def render(self):
        return self.env.render()

# Create and wrap the environment
env = PongEnvWrapper()
check_env(env)  # Validate the environment
env = DummyVecEnv([lambda: env])

# Define hyperparameters
total_timesteps = 1000
eval_interval = 50000
n_eval_episodes = 10

# Initialize the PPO model
model = PPO("MlpPolicy", env, verbose=1, learning_rate=3e-4, n_steps=2048, batch_size=64,
             n_epochs=10, gamma=0.99, gae_lambda=0.95, clip_range=0.2, ent_coef=0.01, vf_coef=0.5,
               max_grad_norm=0.5)


# Check if a pre-trained model exists
if os.path.isfile("ppo_pong_model.zip"):
    print("Loading pre-trained model...")
    model = PPO.load("ppo_pong_model.zip", env=env)
else:
    print("Training new model...")

# Training loop
mean_rewards = []
timesteps = []

for i in range(0, total_timesteps, eval_interval):
    model.learn(total_timesteps=eval_interval, reset_num_timesteps=False)
    
    # Evaluate the model
    mean_reward, _ = evaluate_policy(model, env, n_eval_episodes=n_eval_episodes)
    mean_rewards.append(mean_reward)
    timesteps.append(i + eval_interval)
    
    print(f"Timestep: {i + eval_interval}, Mean Reward: {mean_reward:.2f}")

# Save the trained model
model.save("ppo_pong_model.zip")

# Plot the learning curve
plt.figure(figsize=(10, 5))
plt.plot(timesteps, mean_rewards, marker = 'o')
plt.xlabel("Timesteps")
plt.ylabel("Mean Reward")
plt.title("PPO Learning Curve")
plt.show()

# Final evaluation
mean_reward, std_reward = evaluate_policy(model, env, n_eval_episodes=100)
print(f"Final evaluation - Mean reward: {mean_reward:.2f} +/- {std_reward:.2f}")