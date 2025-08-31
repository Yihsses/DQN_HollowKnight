import torch
import torch.nn as nn
import torch.nn.functional as F

class CombinedNet(nn.Module):
    def __init__(self, num_actions, input_dim):  
        super(CombinedNet, self).__init__()
        self.fc1 = nn.Linear(input_dim, 256)
        self.fc2 = nn.Linear(256, 128)
        self.fc3 = nn.Linear(128, num_actions)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x

# 建立模型
