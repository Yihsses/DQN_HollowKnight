from Tool.screngrap import screngrap
from Tool.action import  take_action
from hollowknight_env import HollowKnightEnv
from Tool.action import restart
from Tool import framebuffer
import torch
from dqnnet import Q_construct
from dqn_3cnn import Q_construct_3d
import time
import numpy as np
# screngrap.grap("HOLLOW KNIGHT")
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
frame_buffer = framebuffer.FrameBuffer(windows_name="HOLLOW KNIGHT", buffer_size=4, capture_interval=0.02)
model =  Q_construct_3d( height=400//4, width=200//4, num_actions=6,image_channels=3).to(device)

frame_buffer.start()

env = HollowKnightEnv()
# take_action(2)
while True:
        frames = frame_buffer.get_latest_3d_frames()
        # state = screngrap.screngrap.grap('HOLLOW KNIGHT')ㄇ
        # state = torch.tensor(state).permute(2, 0, 1)
        # state = torch.tensor(state, dtype=torch.fㄨloat32) / 255.0
        # state = state.unsqueeze(0)
        # action_0 = model.forward(state)
        # rand = np.random.uniform(0, 1)
        rand = np.random.uniform(0, 1)  # 隨機生成一個 0 到 1 之間的數字
        if frames!= None:
            if(frames.shape[2] == 4):
                print("Frames Mean:", frames.mean().item())
                print("Frames Std:", frames.std().item())
                # frames = frames.permute(1, 0, 2, 3).unsqueeze(0)w
                with torch.no_grad(): 
                    action = torch.argmax(model(frames.to(device)), dim=1).item()
                    print("模型" + str(action))
        else:
            # frames = frames.permute(1, 0, 2, 3).unsqueeze(0)
            action = np.random.randint(0, 6)
            time.sleep(0.1)
            # print("隨機：" + str(action))

