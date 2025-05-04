import torch
import torch.nn as nn

import numpy as np
import random
from dqnnet import Q_construct_3d
# from dqnnet import QNetwork
from Tool import framebuffer
from Tool import screngrap
from collections import deque
import time
import matplotlib.pyplot as plt
for i in range (0,10):
    print(np.random.randint(0, 6))
frame_buffer = framebuffer.FrameBuffer(windows_name="HOLLOW KNIGHT", buffer_size=4, capture_interval=0.05)
model = Q_construct_3d(height = 1280//4 ,width=720//4, num_actions=7,image_channels=4)
frame_buffer.start()
while True:
    frames = frame_buffer.get_latest_frames()
    if(frames != None):
        if(len(frames)>=4):
            frames = frames.permute(1, 0, 2, 3).unsqueeze(0)
            action_0 = model(frames)
            action =  torch.argmax(action_0).item()

state = screngrap.screngrap.grap('HOLLOW KNIGHT')
state = torch.tensor(state).permute(2, 0, 1)
state = torch.tensor(state, dtype=torch.float32) / 255.0
state = state.unsqueeze(0)

# epsilon = 0.1
# gridsize = 15
# GAMMA = 0.9
# optimizer = torch.optim.Adam(model.parameters(), lr = 1e-5)


action_0 = model(state)
rand = np.random.uniform(0, 1)