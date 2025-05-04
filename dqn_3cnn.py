import torch
import torch.nn as nn
import torch.nn.functional as F

class QNetwork3D(nn.Module):
    def __init__(self, num_actions, time_steps=4, image_channels=4, height=84, width=84):
        """
        3D CNN Q-Network
        Args:
            num_actions (int): 動作數量。
            time_steps (int): 時間步數（幀數）。
            image_channels (int): 單幀影像的通道數（通常是 RGB，即 3）。
            height (int): 影像高度。
            width (int): 影像寬度。
        """
        super(QNetwork3D, self).__init__()
        
        # 初始參數
        self.time_steps = time_steps
        self.image_channels = image_channels
        self.height = height
        self.width = width
        
        # 3D 卷積層處理時間-空間特徵
        self.conv1 = nn.Conv3d(image_channels, 32, kernel_size=(3, 3, 3), stride=1, padding=1)
        self.conv2 = nn.Conv3d(32, 64, kernel_size=(3, 3, 3), stride=1, padding=1)
        self.pool = nn.MaxPool3d(kernel_size=(2, 2, 2), stride=(2, 2, 2))

        # 計算展平後的大小
        
        # 全連接層
        self.fc1 = nn.Linear(64*height*width, 128)
        self.fc2 = nn.Linear(128, 64)
        self.fc3 = nn.Linear(64, 32)
        self.fc4 = nn.Linear(32, num_actions)
        self.dropout = nn.Dropout(0.5)

        # 使用 Xavier 初始化
        self.init_weights()

    def init_weights(self):
        """
        初始化全連接層的權重。
        """
        nn.init.xavier_uniform_(self.fc1.weight)
        nn.init.xavier_uniform_(self.fc2.weight)
        nn.init.xavier_uniform_(self.fc3.weight)
        nn.init.xavier_uniform_(self.fc4.weight)


    def forward(self, x):
        """
        前向傳播。
        Args:
            x (torch.Tensor): 輸入數據，形狀為 (batch_size, image_channels, time_steps, height, width)。
        Returns:
            torch.Tensor: Q 值，形狀為 (batch_size, num_actions)。
        """
        x = F.relu(self.conv1(x))
        x = self.pool(x)
        x = F.relu(self.conv2(x))
        x = self.pool(x)

        # 展平特徵向量
        batch_size = x.size(0)
        x = x.view(batch_size, -1)

        # 全連接層處理
        x = F.relu(self.fc1(x))
        x = self.dropout(x)
        x = F.relu(self.fc2(x))
        x = self.dropout(x)
        x = F.relu(self.fc3(x))
        x = self.dropout(x)
        x = self.fc4(x)
        return torch.argmax(x, dim=1)

def Q_construct_3d(num_actions, time_steps=4, image_channels=3, height=84, width=84):
    return QNetwork3D(num_actions, time_steps, image_channels, height, width)
