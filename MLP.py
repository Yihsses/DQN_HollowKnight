import torch
import torch.nn as nn
import torch.nn.functional as F

class CoordinateNet(nn.Module):
    def __init__(self, num_actions, input_dim=4):  
        super(CoordinateNet, self).__init__()
        self.fc1 = nn.Linear(input_dim, 128)  # (自己X,Y + BossX,Y)
        self.fc2 = nn.Linear(128, 128)
        self.fc3 = nn.Linear(128, num_actions)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x
