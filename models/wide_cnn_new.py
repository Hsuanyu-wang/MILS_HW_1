import torch
import torch.nn as nn
import torch.nn.functional as F

############## Original ####################
# class WideCNN(nn.Module):
#     def __init__(self, in_channels=3, num_classes=100):
#         super(WideCNN, self).__init__()
#         # 有效層1: 卷積
#         self.conv1 = nn.Conv2d(in_channels, 128, kernel_size=5, stride=1, padding=2)
#         # 有效層2: 卷積
#         self.conv2 = nn.Conv2d(128, 256, kernel_size=3, stride=1, padding=1)
#         # 有效層3: 卷積
#         self.conv3 = nn.Conv2d(256, 512, kernel_size=3, stride=1, padding=1)
#         self.pool = nn.AdaptiveAvgPool2d(1)
#         self.fc = nn.Linear(512, num_classes)

#     def forward(self, x):
#         x = F.relu(self.conv1(x))
#         x = F.relu(self.conv2(x))
#         x = F.relu(self.conv3(x))
#         x = self.pool(x)
#         x = x.view(x.size(0), -1)
#         x = self.fc(x)
#         return x

# Wider
# class WideCNN(nn.Module):
#     def __init__(self, in_channels=3, num_classes=100):
#         super(WideCNN, self).__init__()
#         # 有效層1: 卷積
#         # self.conv1 = nn.Conv2d(in_channels, 128, kernel_size=5, stride=1, padding=2)
#         self.conv1 = nn.Conv2d(in_channels, 512, kernel_size=7, stride=1, padding=3)
#         self.bn1 = nn.BatchNorm2d(512)
#         # 有效層2: 卷積
#         # self.conv2 = nn.Conv2d(128, 256, kernel_size=3, stride=1, padding=1)
#         self.conv2 = nn.Conv2d(512, 512, kernel_size=5, stride=1, padding=2)
#         self.bn2 = nn.BatchNorm2d(512)
#         # 有效層3: 卷積
#         # self.conv3 = nn.Conv2d(256, 512, kernel_size=3, stride=1, padding=1)
#         self.conv3 = nn.Conv2d(512, 1024, kernel_size=3, stride=1, padding=1)
#         self.bn3 = nn.BatchNorm2d(1024)
#         self.pool = nn.AdaptiveAvgPool2d(1)
#         #
#         self.dropout = nn.Dropout(0.5)
#         #
#         # self.fc = nn.Linear(512, num_classes)
#         self.fc = nn.Linear(1024, num_classes)

#     def forward(self, x):
#         # x = F.relu(self.conv1(x))
#         x = F.relu(self.bn1(self.conv1(x)))
#         # x = F.relu(self.conv2(x))
#         x = F.relu(self.bn2(self.conv2(x)))
#         # x = F.relu(self.conv3(x))
#         x = F.relu(self.bn3(self.conv3(x)))
#         x = self.pool(x)
#         x = x.view(x.size(0), -1)
#         #
#         x = self.dropout(x)
#         # 
#         x = self.fc(x)
#         return x


# With Residual block
# class ResidualBlock(nn.Module):
#     def __init__(self, in_channels, out_channels, stride=1):
#         super(ResidualBlock, self).__init__()
#         self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=stride, padding=1, bias=False)
#         self.bn1 = nn.BatchNorm2d(out_channels)
#         self.shortcut = nn.Sequential()
#         if stride != 1 or in_channels != out_channels:
#             self.shortcut = nn.Sequential(
#                 nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=stride, bias=False),
#                 nn.BatchNorm2d(out_channels)
#             )

#     def forward(self, x):
#         out = F.relu(self.bn1(self.conv1(x)))
#         out = out + self.shortcut(x)  # 改成 out-of-place
#         out = F.relu(out)
#         return out

# class Residual_WideCNN(nn.Module):
#     def __init__(self, in_channels=3, num_classes=100):
#         super(Residual_WideCNN, self).__init__()
#         # 有效層1: 卷積
#         self.conv1 = nn.Conv2d(in_channels, 128, kernel_size=5, stride=1, padding=2)
#         # 有效層2: 殘差塊（只含一層卷積）
#         self.resblock1 = ResidualBlock(128, 256)
#         # 有效層3: 殘差塊（只含一層卷積）
#         self.resblock2 = ResidualBlock(256, 512)
#         self.pool = nn.AdaptiveAvgPool2d(1)
#         self.fc = nn.Linear(512, num_classes)

#     def forward(self, x):
#         x = F.relu(self.conv1(x))
#         x = self.resblock1(x)
#         x = self.resblock2(x)
#         x = self.pool(x)
#         x = x.view(x.size(0), -1)
#         x = self.fc(x)
#         return x
    
class MultiResidualBlock(nn.Module):
    def __init__(self, in_channels, out_channels, stride=1):
        super().__init__()
        # 分支1
        self.conv1a = nn.Conv2d(in_channels, out_channels, 3, stride, 1, bias=False)
        self.bn1a = nn.BatchNorm2d(out_channels)
        # 分支2
        self.conv1b = nn.Conv2d(in_channels, out_channels, 3, stride, 1, bias=False)
        self.bn1b = nn.BatchNorm2d(out_channels)
        # shortcut
        self.shortcut = nn.Sequential()
        if stride != 1 or in_channels != out_channels:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, 1, stride, bias=False),
                nn.BatchNorm2d(out_channels)
            )
    def forward(self, x):
        out_a = F.relu(self.bn1a(self.conv1a(x)))
        out_b = F.relu(self.bn1b(self.conv1b(x)))
        out = out_a + out_b + self.shortcut(x)
        return F.relu(out)

class Residual_WideCNN(nn.Module):
    def __init__(self, in_channels=3, num_classes=100):
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels, 128, kernel_size=5, stride=1, padding=2)
        self.bn1 = nn.BatchNorm2d(128)
        self.block1 = MultiResidualBlock(128, 256, stride=2)
        self.block2 = MultiResidualBlock(256, 512, stride=2)
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.dropout = nn.Dropout(0.5)
        self.fc = nn.Linear(512, num_classes)
    def forward(self, x):
        x = F.relu(self.bn1(self.conv1(x)))
        x = self.block1(x)
        x = self.block2(x)
        x = self.pool(x)
        x = x.view(x.size(0), -1)
        x = self.dropout(x)
        x = self.fc(x)
        return x