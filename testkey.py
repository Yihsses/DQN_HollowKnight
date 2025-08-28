import torch
print("CUDA 是否可用：", torch.cuda.is_available())
print("CUDA 裝置數量：", torch.cuda.device_count())
print("目前使用的 GPU 名稱：", torch.cuda.get_device_name(0) if torch.cuda.is_available() else "無")