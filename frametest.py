import cv2
import threading
import time
import collections
import numpy as np
import torch
import torch.nn as nn
from dqnnet import Q_construct
from Tool import framebuffer
# 初始化 FrameBuffer

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
frame_buffer = framebuffer.FrameBuffer(windows_name="HOLLOW KNIGHT", buffer_size=4, capture_interval=0.05)
model =  Q_construct(input_dim=int((400/4)*(200/4)), num_actions=6,image_channels=4).to(device)
# 啟動 FrameBuffer 執行緒
frame_buffer.start()
optimizer = torch.optim.Adam(model.parameters(), lr = 1e-5)


try:

    while True:
        # 獲取最新的 4 幀影像
        frames = frame_buffer.get_latest_frames()
        if(len(frames)>=4):
            epsilon = 0.1
            gridsize = 15
            GAMMA = 0.9

            action_0 = model(frames)
            print
            rand = np.random.uniform(0, 1)
        if frames is not None:
            # 將 4 幀影像合併為單張大圖進行顯示（僅用於測試）
            

            print()

        # 按下 'q' 鍵退出


except KeyboardInterrupt:
    print("停止中...")

finally:
    # 停止 FrameBuffer 並釋放資源
    frame_buffer.stop()
    frame_buffer.join()
