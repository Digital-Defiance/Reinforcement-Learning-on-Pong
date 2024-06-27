import torch
import random
from collections import deque
import numpy as np
import pickle


class ReplayBuffer():
    def __init__(self):
        self.buffer_size = int(1e6)
        self.batch_size = 32
        self.index = 0
        self.size = 0

        self.states = torch.empty((self.buffer_size, 6))
        self.actions = torch.empty((self.buffer_size, 1))
        self.rewards = torch.empty((self.buffer_size, 1))
        self.next_states = torch.empty((self.buffer_size, 6))
        self.dones = torch.empty((self.buffer_size, 1))
        

    def __len__(self):
        return self.size

    def add(self, s, a, r, ns, d):
        self.states[self.index] = torch.tensor(s, dtype=torch.float32)
        self.actions[self.index] = a
        self.rewards[self.index] = r
        self.next_states[self.index] = torch.tensor(ns, dtype=torch.float32)
        self.dones[self.index] = d

        self.index = (self.index + 1) % self.buffer_size
        self.size = min(self.size + 1, self.buffer_size)
        
    def sample_batch(self):
        indices = torch.randint(0, self.size, (self.batch_size, ))

        return (
            self.states[indices],
            self.actions[indices],
            self.rewards[indices],
            self.next_states[indices],
            self.dones[indices]
        )

    def save_buffer(self, filepath = "buffer.pkl"):
        buffer_data = {
            'states' : self.states[:self.size],
            'actions' : self.actions[:self.size],
            'rewards' : self.rewards[:self.size],
            'next_states' : self.next_states[:self.size],
            'dones' : self.dones[:self.size],
            'size' : self.size,
            'indes' : self.index
        }

        with open(filepath, 'wb') as f:
            pickle.dump(buffer_data, f)
    
    def load_buffer(self, filepath = "buffer.pkl"):
        with open(filepath, 'rb') as f:
            buffer_data = pickle.load(f)

        self.size = buffer_data['size']
        self.index = buffer_data['index']

        if self.size > self.buffer_size:
            self.buffer_size = self.size
            self.states = torch.empty((self.buffer_size, 6))
            self.actions = torch.empty((self.buffer_size, 1))
            self.rewards = torch.empty((self.buffer_size, 1))
            self.next_states = torch.empty((self.buffer_size, 6))
            self.dones = torch.empty((self.buffer_size, 1))

        self.states[:self.size] = buffer_data['states']
        self.actions[:self.size] = buffer_data['actions']
        self.rewards[:self.size] = buffer_data['rewards']
        self.next_states[:self.size] = buffer_data['next_states']
        self.dones[:self.size] = buffer_data['dones']

        


