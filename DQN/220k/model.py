import torch
import torch.nn.functional as F


class DQN(torch.nn.Module):

    def __init__(self, in_features):
        super(DQN, self).__init__()
        self.fc1 = torch.nn.Linear(in_features, 128)
        self.fc2 = torch.nn.Linear(128, 64)
        self.fc3 = torch.nn.Linear(64, 2)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        return self.fc3(x)
    