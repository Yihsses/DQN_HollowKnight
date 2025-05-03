import torch
import torch.nn as nn

import numpy as np
import random
from dqnnet import Q_construct
from dqnnet import QNetwork

from Tool import screngrap
from collections import deque
import time
import matplotlib.pyplot as plt

state = screngrap.screngrap.grap('HOLLOW KNIGHT')
state = torch.tensor(state).permute(2, 0, 1)
state = torch.tensor(state, dtype=torch.float32) / 255.0
state = state.unsqueeze(0)
model = Q_construct(input_dim=int((1280/4)*(720/4)), num_actions=7,image_channels=4)
epsilon = 0.1
gridsize = 15
GAMMA = 0.9
optimizer = torch.optim.Adam(model.parameters(), lr = 1e-5)


action_0 = model(state)
rand = np.random.uniform(0, 1)