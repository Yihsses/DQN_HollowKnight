import torch
import torch.nn as nn
from torchvision import models

class ResNetEmbedding(nn.Module):
    def __init__(self, model_path: str, device: torch.device):
        super(ResNetEmbedding, self).__init__()
        self.device = device
        
        # 建立 ResNet18
        self.resnet = models.resnet18(pretrained=False)
        self.resnet.fc = nn.Identity()  # 移除最後分類層，輸出 embedding
        self.resnet.load_state_dict(torch.load(model_path, map_location=device))
        self.resnet.to(device)
        self.resnet.eval()  # 推理模式
        
        self.embedding_dim = 512  # ResNet18最後全連接層輸入維度

    def forward(self, x):
        """
        x: [B, C, H, W] tensor
        return: [B, embedding_dim] tensor
        """
        with torch.no_grad():
            embedding = self.resnet(x.to(self.device))
        return embedding
