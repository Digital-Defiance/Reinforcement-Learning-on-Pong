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

# Helper functions
# why this function -> for training DQN model using mini-batch of experince tuple from replay buffer

def update_dqn(model, target_model, optimizer, batch, gamma, losses):
    # aip just unpacks the batch into separate tuple
    states, actions, rewards, next_states, done = zip(*batch)
    # now creating pytorch tensors
    
    states = np.array(states, dtype=np.float32)
    actions = np.array(actions, dtype=np.int64)
    rewards = np.array(rewards, dtype=np.float32)
    next_states = np.array(next_states, dtype=np.float32)
    dones = np.array(done, dtype=np.float32)

    states = torch.tensor(states, dtype=torch.float32)
    actions = torch.tensor(actions, dtype=torch.int64)
    rewards = torch.tensor(rewards, dtype=torch.float32)
    next_states = torch.tensor(next_states, dtype=torch.float32)
    dones = torch.tensor(done, dtype=torch.float32)


    states = states.view(states.size(0), -1)
    next_states = next_states.view(next_states.size(0), -1)

    q_values = model(states)
    next_q_values = target_model(next_states).max(dim = 1)[0].detach()


    # now here is the formula
    target_q_values = rewards + gamma * next_q_values * (1 - dones)

    # calculate loss -> smooth l1 loss is advance of l1_loss ( MAE )
    # the loss of smooth_l1_loss approaches to 0
    loss = F.smooth_l1_loss(q_values.gather(dim = 1, index = actions.unsqueeze(-1)), target_q_values.unsqueeze(-1))
    print("current loss -> ", loss)
    # what is unsqueeze = -1, it adds new dimension at last

    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    
    return loss

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
batch_size = 64
replay_buffer_size = int(1e6) # 1e6 for Atari and 1e5 for classic games
num_episodes = 2
losses = []

# Initializing environments now!
env = PongEnvironment()
state_size = 6 # striker_y, striker_x, ball_y, ball_X, velocity of ball (x and y) 
action_size = 2 # left and right

model = DQN(state_size)
target_model = DQN(state_size)
optimizer = optim.Adam(model.parameters(), lr = learning_rate)

if os.path.isfile("trained_model.pth"):
    checkpoint = torch.load("trained_model.pth")
    model.load_state_dict(checkpoint['model_state_dict'])
    optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    losses.extend(checkpoint['losses'])
    total_num_episodes = checkpoint['total_num_episodes']
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
    episode_experiences = []
    done = False
    state = env.get_striker_and_ball_coordinates()

    while not done:
        action = epsilon_greedy_policy(state, epsilon, model, action_size)
        _, reward, done, _ = env.step(action)
        next_state = env.get_striker_and_ball_coordinates()
        # replay_buffer.append((state, action, reward, next_state, done))
        episode_experiences.append((state, action, reward, next_state, done))
        state = next_state
        
        if reward == 10 or reward == -10:
            done = True

        if episode % target_update_freq == 0:
            target_model.load_state_dict(model.state_dict())


        display.clear_output(wait = True)
        # env.render()
    

    # batch = random.sample(replay_buffer, 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batch_size)
    loss = update_dqn(model, target_model, optimizer, episode_experiences, gamma, losses)
    losses.append(loss.item())
    # Update the target model

    epsilon = max(epsilon_end, epsilon * 0.995)
    print(f"Episode {episode + 1} : Total Reward = {reward}")

print("---------------")
print("Total reward -> ", total_reward)



# now saving model using checkpoints
PATH = "trained_model.pth"
LOSSES = []
LOSSES.extend(losses)
TOTAL_NUM_EPISODES = num_episodes + total_num_episodes

torch.save({
    'model_state_dict' : model.state_dict(),
    'optimizer_state_dict' : optimizer.state_dict(),
    'losses' : LOSSES,   
    'total_num_episodes' : TOTAL_NUM_EPISODES
}, PATH)

plt.plot(LOSSES)
plt.show()