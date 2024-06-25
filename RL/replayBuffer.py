import torch
import random
from collections import deque
import numpy as np
import pickle


class ReplayBuffer():
    def __init__(self):
        pass        

    def __len__(self):
        return len(self.buffer)
    
    def create_buffer(self, buffer_size):
        self.buffer_size = buffer_size
        self.buffer = deque(maxlen=self.buffer_size)
    
    def append(self, experience):
        self.buffer.append(experience)

    def return_tensor(self, episode_experiences):
        self.append(episode_experiences)
        states, actions, rewards, next_states, done = zip(*episode_experiences)

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

        return states, actions, rewards, next_states, dones

    def save_buffer(self):
        with open("buffer.pkl", 'wb') as f:
            pickle.dump(list(self.buffer), f)
    
    def load_buffer(self):
        with open("buffer.pkl", 'rb') as f:
            self.buffer = deque(pickle.load(f))


