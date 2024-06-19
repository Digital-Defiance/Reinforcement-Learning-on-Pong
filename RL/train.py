import gym
import numpy as np
import torch
import torch.optim as optim
import torch.nn.functional as F
from collections import deque
from model import DQN
from environment import PongEnvironment


# defining hyperparameters
gamma = 0.99 # Discount factor
epilson_start = 1.0 # this value will get reduce 
epilson_end = 0.01 # till this our model will know path
target_update_freq = 1000
learning_rate = 0.001
batch_size = 64
replay_buffer_size = 10000
num_episodes = 100


# Initializing environments now!
env = PongEnvironment()

state_size = env.observation_space.shape[0]
action_size = 2 # left and right


model = DQN(state_size, action_size)
