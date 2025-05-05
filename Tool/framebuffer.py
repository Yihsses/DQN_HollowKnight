import threading
import time
import collections
import numpy as np
import torch
from Tool import screngrap

class FrameBuffer(threading.Thread):
    def __init__(self, windows_name, buffer_size=4, capture_interval=0.05):
        super(FrameBuffer, self).__init__()
        self.deivce = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.windows_name = windows_name
        self.buffer_size = buffer_size
        self.capture_interval = capture_interval
        self.buffer = collections.deque(maxlen=buffer_size)
        self.running = False

    def preprocess_frame(self, frame):
        """
        對影像進行預處理：
        - 轉換為 PyTorch 張量
        - 通道順序變更為 (C, H, W)
        - 歸一化至 [0, 1]
        - 添加批次維度
        """
        #  state = torch.tensor(frame).permute(2, 0, 1)  # 調整通道順序
        state = torch.tensor(frame, dtype=torch.float32) / 255.0  # 歸一化
        state = state.unsqueeze(0)  # 添加批次維度
        state = state.unsqueeze(0)
        return state

    def run(self):
        """啟動背景執行緒，連續抓取影像。"""
        self.running = True
        while self.running:
            # 抓取單幀影像
            frame = screngrap.screngrap.grap(self.windows_name)
            preprocessed_frame = self.preprocess_frame(frame)  # 預處理影像
            self.buffer.append(preprocessed_frame)  # 保存到緩衝區
            time.sleep(self.capture_interval)  # 等待間隔

    def stop(self):
        """停止抓取執行緒。"""
        self.running = False

    def get_latest_frames(self):
        """
        獲取最新的 4 幀影像。

        返回:
        - 如果緩衝區不足 4 幀，返回 None。
        - 如果緩衝區有足夠的影像，返回形狀為 (4, C, H, W) 的 PyTorch 張量。
        """
        start_time = time.time()  # 開始計時
        if len(self.buffer) < self.buffer_size:
            return None  # 不足 4 幀時返回 None
        result = torch.cat(list(self.buffer), dim=0)  # 堆疊成多幀影像
        end_time = time.time()  # 結束計時
        print(f"處理時間: {end_time - start_time:.6f} 秒")  # 輸出處理時間
        return result