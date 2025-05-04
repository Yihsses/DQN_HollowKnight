import torch
import torch.nn as nn
import numpy as np
from dqn_3cnn import Q_construct_3d
from Tool import framebuffer
from Tool import screngrap

# 檢查 GPU
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# 初始化 FrameBuffer 和模型
frame_buffer = framebuffer.FrameBuffer(windows_name="HOLLOW KNIGHT", buffer_size=4, capture_interval=0.05)
model = Q_construct_3d(height=1280 // 4, width=720 // 4, num_actions=7, image_channels=4).to(device)
frame_buffer.start()

while True:
    frames = frame_buffer.get_latest_frames()
    if frames is not None:
        if len(frames) >= 4:
            # 調整維度並移動到 GPU
            frames = frames.permute(1, 0, 2, 3).unsqueeze(0).to(device)
            
            # 使用模型進行推論
            with torch.no_grad():  # 關閉梯度計算以加速推論
                action_0 = model(frames)
                print(action_0)

# 抓取單幀並移動到 GPU
state = screngrap.screngrap.grap('HOLLOW KNIGHT')
state = torch.tensor(state).permute(2, 0, 1)
state = torch.tensor(state, dtype=torch.float32) / 255.0
state = state.unsqueeze(0).to(device)

# 推論動作
with torch.no_grad():
    action_0 = model(state)
rand = np.random.uniform(0, 1)
