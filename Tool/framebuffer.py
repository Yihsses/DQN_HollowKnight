import threading
import time
import collections
import numpy as np
import torch
from Tool import screngrap

class FrameBuffer(threading.Thread):
    def __init__(self, windows_name, buffer_size=4, capture_interval=0.1):
        super(FrameBuffer, self).__init__()
        self.windows_name = windows_name
        self.buffer_size = buffer_size
        self.capture_interval = capture_interval
        self.buffer = collections.deque(maxlen=buffer_size)
        self.running = False

    def preprocess_frame(self, frame):
        # state = torch.tensor(frame).unsqueeze(0)  # 調整通道順序
        # state = torch.tensor(frame, dtype=torch.float32) / 255.0 # 歸一化
        # state = state.unsqueeze(0)  # 添加批次維度

        state = torch.tensor(frame, dtype=torch.float32)   # 歸一化
        state = state.unsqueeze(0)  # 添加批次維度
        state = state.unsqueeze(0) 
        # return state
        return state

    def run(self):
        """啟動背景執行緒，連續抓取影像。"""
        self.running = True
        while self.running:
            # 抓取單幀影像
            frame = screngrap.screngrap.grap(self.windows_name)
            preprocessed_frame = self.preprocess_frame(frame)  # 預處理影像d
            self.buffer.append(preprocessed_frame)  # 保存到緩衝區
            time.sleep(self.capture_interval)  # 等待間隔

    def stop(self):
        """停止抓取執行緒。"""
        self.running = False

    def get_latest_frames(self):
        if len(self.buffer) < self.buffer_size:
            return None  # 不足 4 幀時返回 None
        
        # 堆疊成多幀影像 (4, 1, 200, 400)
 # [4, 1, 200, 400]
        stacked_frames = torch.stack(list(self.buffer), dim=0)
        stacked_frames = stacked_frames.view(1, -1, stacked_frames.shape[1], stacked_frames.shape[2])  # [1, 12, 400, 200]
        # 重新排列維度，形成 (1, 4, 200, 400)
     # [1, 4, 200, 400]
        return stacked_frames
    def get_latest_3d_frames(self):

        if len(self.buffer) < self.buffer_size:
            return None  # 不足 4 幀時返回 None
        # permute(3, 0, 1, 2)
        stacked_frames = torch.cat(list(self.buffer), dim=0)
        stacked_frames = stacked_frames.permute(1,0,2,3).unsqueeze(0)  
        return stacked_frames
