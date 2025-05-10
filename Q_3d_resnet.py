import torch
import torch.nn as nn
import torch.nn.functional as F

class BasicBlock3D(nn.Module):
    def __init__(self, in_channels, out_channels, stride=1):
        super(BasicBlock3D, self).__init__()
        self.conv1 = nn.Conv3d(in_channels, out_channels, kernel_size=3, stride=stride, padding=1)
        self.bn1 = nn.BatchNorm3d(out_channels)
        self.conv2 = nn.Conv3d(out_channels, out_channels, kernel_size=3, stride=1, padding=1)
        self.bn2 = nn.BatchNorm3d(out_channels)
        self.shortcut = nn.Sequential()
        if stride != 1 or in_channels != out_channels:
            self.shortcut = nn.Sequential(
                nn.Conv3d(in_channels, out_channels, kernel_size=1, stride=stride),
                nn.BatchNorm3d(out_channels)
            )

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out += self.shortcut(x)
        return F.relu(out)


class ResNet3D(nn.Module):
    def __init__(self, num_actions, image_channels=3, time_steps=4, height=400, width=200):
        super(ResNet3D, self).__init__()
        self.conv1 = nn.Conv3d(image_channels, 32, kernel_size=(2, 3, 3), stride=(1, 2, 2), padding=(0, 1, 1))
        self.bn1 = nn.BatchNorm3d(32)

        self.conv2 = nn.Conv3d(32, 48, kernel_size=(2, 3, 3), stride=1, padding=(0, 1, 1))
        self.bn2 = nn.BatchNorm3d(48)

        self.conv3 = nn.Conv3d(48, 64, kernel_size=(2, 3, 3), stride=1, padding=(0, 1, 1))
        self.bn3 = nn.BatchNorm3d(64)

        self.layer1 = self._make_layer(BasicBlock3D, 64, 96, 2, stride=2)
        self.layer2 = self._make_layer(BasicBlock3D, 96, 128, 2, stride=2)
        self.layer3 = self._make_layer(BasicBlock3D, 128, 256, 2, stride=2)

        self.global_pool = nn.AdaptiveAvgPool3d(1)
        self.fc1 = nn.Linear(256, 128)
        self.fc2 = nn.Linear(128, 64)
        self.fc3 = nn.Linear(64, num_actions)
    def _make_layer(self, block, in_channels, out_channels, num_blocks, stride):
        layers = []
        layers.append(block(in_channels, out_channels, stride))
        for _ in range(1, num_blocks):
            layers.append(block(out_channels, out_channels))
        return nn.Sequential(*layers)

    def forward(self, x):
        x = F.relu(self.bn1(self.conv1(x)))
        x = F.relu(self.bn2(self.conv2(x)))
        x = F.relu(self.bn3(self.conv3(x)))

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)

        x = self.global_pool(x)
        x = torch.flatten(x, 1)
        x = self.fc1(x)
        x = self.fc2(x)
        x = self.fc3(x)
        return x

# Example instantiation

