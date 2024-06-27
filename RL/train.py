import gym
import numpy as np
import torch
import torch.optim as optim
import torch.nn.functional as F
from collections import deque
from model import DQN
from environment import PongEnvironment
import random
from IPython import display
import matplotlib.pyplot as plt
import os
from replayBuffer import ReplayBuffer
import time

# Helper functions
# why this function -> for training DQN model using mini-batch of experince tuple from replay buffer

def update_dqn(model, target_model, optimizer, batch, replay_buffer, gamma, losses):
    
    states, actions, rewards, next_states, dones = batch
    actions = actions.long()
   
    states = states.view(states.size(0), -1)
    next_states = next_states.view(next_states.size(0), -1)

    q_values = model(states).gather(1, actions.view(-1, 1)).squeeze(1)
    next_q_values = target_model(next_states).max(dim = 1)[0].detach()


    # now here is the formula
    target_q_values = rewards + gamma * next_q_values * (1 - dones)

    loss = F.mse_loss(q_values, target_q_values.detach())
    # print("current loss -> ", loss)
    # what is unsqueeze = -1, it adds new dimension at last

    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    
    return loss, replay_buffer

def epsilon_greedy_policy(state, epsilon, model, action_size):
    if np.random.rand() < epsilon:
        return np.random.choice(action_size)

    else:
        with torch.no_grad():
            state = torch.tensor(state, dtype=torch.float32).view(1, -1)
            q_values = model(state)
            return q_values.argmax().item()

# defining hyperparameters
gamma = 0.99 # Discount factor
epsilon_start = 1.0 # this value will get reduce 
epsilon_end = 0.01 # till this our model will know path
target_update_freq = 1000
learning_rate = 0.001
num_episodes = 10
losses = []

# Initializing environments now!
env = PongEnvironment()
replay_buffer = ReplayBuffer()
state_size = 6 # striker_y, striker_x, ball_y, ball_X, velocity of ball (x and y) 
action_size = 2 # left and right
batch_size = 32 # batch size for replay_buffer
model = DQN(state_size)
target_model = DQN(state_size)
optimizer = optim.Adam(model.parameters(), lr = learning_rate)

if os.path.isfile("trained_model.pth"):
    checkpoint = torch.load("trained_model.pth")
    model.load_state_dict(checkpoint['model_state_dict'])
    optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    losses.extend(checkpoint['losses'])
    total_num_episodes = checkpoint['total_num_episodes']
    if os.path.isfile('buffer.pkl'):
        replay_buffer.load_buffer()
    model.train()

else:
    print("TRAINING FOR FIRST TIME!")  
    total_num_episodes = 0
    model.train()

target_model.load_state_dict(model.state_dict())
target_model.eval()
# why eval -> to stop dropout layers, batch normalization


# Training Loop
epsilon = epsilon_start
total_reward = 0
for episode in range(num_episodes):
    env.reset()
    done = False
    state = env.get_striker_and_ball_coordinates()


    while not done:
        action = epsilon_greedy_policy(state, epsilon, model, action_size)
        _, reward, done, _ = env.step(action)
        next_state = env.get_striker_and_ball_coordinates()
        replay_buffer.add(state, action, reward, next_state, done)
        state = next_state
        
        if len(replay_buffer) >= batch_size:
            batch = replay_buffer.sample_batch()
            loss, replay_buffer = update_dqn(model, target_model, optimizer, batch, replay_buffer, gamma, losses)
            losses.append(loss.item())

        if reward == 1 or reward == -1:
            done = True

        if episode % target_update_freq == 0:
            target_model.load_state_dict(model.state_dict())

        display.clear_output(wait = True)
        # env.render()
        

    epsilon = max(epsilon_end, epsilon * 0.995)
    print(f"Episode {episode + 1} : Reward = {reward}")

print("---------------")
print("Total reward -> ", total_reward)



# now saving model using checkpoints
PATH = "trained_model.pth"
LOSSES = []
LOSSES.extend(losses)
TOTAL_NUM_EPISODES = num_episodes + total_num_episodes

replay_buffer.save_buffer()
torch.save({
    'model_state_dict' : model.state_dict(),
    'optimizer_state_dict' : optimizer.state_dict(),
    'losses' : LOSSES,   
    'total_num_episodes' : TOTAL_NUM_EPISODES
}, PATH)

print("Total number of episode on which model is trained on -> ", TOTAL_NUM_EPISODES)
plt.plot(LOSSES)
# plt.xticks(range(1, TOTAL_NUM_EPISODES), labels=None)
plt.yscale("log")
plt.show()