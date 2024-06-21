import gym
import numpy as np
import torch
import torch.optim as optim
import torch.nn.functional as F
from collections import deque
from model import DQN
from environment import PongEnvironment
import random

# defining hyperparameters
gamma = 0.99 # Discount factor
epsilon_start = 1.0 # this value will get reduce 
epsilon_end = 0.01 # till this our model will know path
target_update_freq = 1000
learning_rate = 0.001
batch_size = 64
replay_buffer_size = 10000
num_episodes = 2


# Initializing environments now!
env = PongEnvironment()
state_size = env.observation_space.shape[0] * env.observation_space.shape[1] * env.observation_space.shape[2]
action_size = 2 # left and right


model = DQN(state_size)
target_model = DQN(state_size)

target_model.load_state_dict(model.state_dict())
target_model.eval()
# why eval -> to stop dropout layers, batch normalization

optimizer = optim.Adam(model.parameters(), lr = learning_rate)

# deque for storing tupele of (state, action, reward, next_state, done) during training
replay_buffer = deque(maxlen = replay_buffer_size)



# Helper functions

# why this function -> for training DQN model using mini-batch of experince tuple from replay buffer

def update_dqn(model, target_model, optimizer, batch, gamma):
    # aip just unpacks the batch into separate tuple
    states, actions, rewards, next_states, done = zip(*batch)

    # now creating pytorch tensors
    states = torch.tensor(states, dtype=torch.float32)
    actions = torch.tensor(actions, dtype=torch.int64)
    rewards = torch.tensor(rewards, dtype=torch.float32)
    next_states = torch.tensor(next_states, dtype=torch.float32)
    dones = torch.tensor(done, dtype=torch.float32)

    q_values = model(states)
    next_q_values = target_model(next_states).max(dim = 1)[0].detach()

    # now here is the formula
    target_q_values = rewards + gamma * next_q_values * (1 - dones)

    # calculate loss -> smooth l1 loss is advance of l1_loss ( MAE )
    # the loss of smooth_l1_loss approaches to 0
    loss = F.smooth_l1_loss(q_values.gather(dim = 1, index = actions.unsqueeze(-1)), target_q_values.unsqueeze(-1))
    # what is unsqueeze = -1, it adds new dimension at last

    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

def epsilon_greedy_policy(state, epsilon, model, action_size):
    if np.random.rand() < epsilon:
        return np.random.choice(action_size)

    else:
        with torch.no_grad():
            q_values = model(torch.tensor(state, dtype=torch.float32))
            return q_values.argmax().item()
        


# Training Loop

epsilon = epsilon_start
for episode in range(num_episodes):
    state = env.reset()
    total_reward = 0
    done = False
    render_interval = 10

    while not done:
        action = epsilon_greedy_policy(state, epsilon, model, action_size)
        next_state, reward, done, _ = env.step(action)

        replay_buffer.append((state, action, reward, next_state, done))
        total_reward += reward
        state = next_state

        if len(replay_buffer) > batch_size:
            batch = random.sample(replay_buffer, batch_size)
            update_dqn(model, target_model, optimizer, batch, gamma)

        if episode % target_update_freq == 0:
            target_model.load_state_dict(model.state_dict())

        if episode % render_interval == 0:
            env.render()


    epsilon = max(epsilon_end, epsilon * 0.995)
    print(f"Epsilon {episode + 1} : Total Reward = {total_reward}")


torch.save(model.state_dict(), "trained_model.pth")
