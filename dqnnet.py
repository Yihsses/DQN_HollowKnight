import torch
import torch.nn as nn
import torch.nn.functional as F

class QNetwork(nn.Module):
    def __init__(self, input_dim, num_actions, image_channels=4):  # 將 image_channels 預設為 4
        super(QNetwork, self).__init__()
        
        # 卷積層處理圖片特徵
        self.conv1 = nn.Conv2d(image_channels, 32, kernel_size=3, stride=1, padding=1)  # 修改 in_channels
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1)
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)

        # 計算卷積後的特徵向量大小
        conv_output_size = 64

        # 全連接層處理非圖像輸入
        self.fc1 = nn.Linear(input_dim * conv_output_size, 128)
        self.fc2 = nn.Linear(128, 64)
        self.fc3 = nn.Linear(64, 32)
        self.fc4 = nn.Linear(32, num_actions)
        self.dropout = nn.Dropout(0.5)
        
        # 使用 Xavier 初始化權重
        nn.init.xavier_uniform_(self.fc1.weight)
        nn.init.xavier_uniform_(self.fc2.weight)
        nn.init.xavier_uniform_(self.fc3.weight)
        nn.init.xavier_uniform_(self.fc4.weight)

    def forward(self, x):
        # 圖片部分的特徵提取
        img_feat = F.relu(self.conv1(x))
        img_feat = self.pool(img_feat)
        img_feat = F.relu(self.conv2(img_feat))
        img_feat = self.pool(img_feat)
        
        batch_size = img_feat.size(0)
        img_feat = img_feat.view(batch_size, -1)
        x = F.relu(self.fc1(img_feat))
        # x = self.dropout(x)
        x = F.relu(self.fc2(x))
        # x = self.dropout(x)
        x = F.relu(self.fc3(x))
        # x = self.dropout(x)
        x = self.fc4(x)
        return x


def Q_construct(input_dim, num_actions, image_channels=4):
    return QNetwork(input_dim=input_dim, num_actions=num_actions, image_channels=image_channels)