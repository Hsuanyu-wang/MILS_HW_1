import torch
import torch.nn as nn
import torch.nn.functional as F

########################################################################################
# Task A
########################################################################################

# class Attention2d(nn.Module):
#     def __init__(self, in_planes, K):
#         super(Attention2d, self).__init__()
#         self.avgpool = nn.AdaptiveAvgPool2d(1)
#         self.fc1 = nn.Conv2d(in_planes, K, 1)
#         self.fc2 = nn.Conv2d(K, K, 1)

#     def forward(self, x):
#         x = self.avgpool(x)
#         x = self.fc1(x)
#         x = F.relu(x)
#         x = self.fc2(x).view(x.size(0), -1)
#         return F.softmax(x, 1)

# class DynamicConv2d(nn.Module):
#     def __init__(self, in_planes, out_planes, kernel_size, stride=1, padding=0, dilation=1, groups=1, bias=True, K=4):
#         super(DynamicConv2d, self).__init__()
#         self.in_planes = in_planes
#         self.out_planes = out_planes
#         self.kernel_size = kernel_size
#         self.stride = stride
#         self.padding = padding
#         self.dilation = dilation
#         self.groups = groups
#         self.bias = bias
#         self.K = K

#         # K組卷積核
#         self.weight = nn.Parameter(torch.randn(K, out_planes, in_planes, kernel_size, kernel_size))
#         if bias:
#             self.bias_param = nn.Parameter(torch.randn(K, out_planes))
#         else:
#             self.bias_param = None

#         self.attention = Attention2d(in_planes, K)

#     def forward(self, x):
#         batch_size = x.size(0)
#         attention_weights = self.attention(x)  # (batch, K)

#         # 動態聚合K組卷積核
#         weight = torch.sum(
#             attention_weights.view(batch_size, self.K, 1, 1, 1, 1) * self.weight, dim=1
#         )  # (batch, out_planes, in_planes, k, k)
#         if self.bias:
#             bias = torch.sum(
#                 attention_weights.view(batch_size, self.K, 1) * self.bias_param, dim=1
#             )  # (batch, out_planes)
#         else:
#             bias = None

#         # 對每個樣本分別做卷積
#         outputs = []
#         for i in range(batch_size):
#             outputs.append(
#                 F.conv2d(
#                     x[i:i+1], weight[i], bias[i] if bias is not None else None,
#                     self.stride, self.padding, self.dilation, self.groups
#                 )
#             )
#         return torch.cat(outputs, dim=0)

class BasicBlock(nn.Module):
    expansion = 1
    def __init__(self, in_channels, out_channels, stride=1, downsample=None):
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, 3, stride, 1, bias=False)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(out_channels, out_channels, 3, 1, 1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_channels)
        self.downsample = downsample

    def forward(self, x):
        identity = x
        out = self.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        if self.downsample:
            identity = self.downsample(x)
        out += identity
        return self.relu(out)

class DynamicConv(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1, padding=1, hidden_dim=64):
        super(DynamicConv, self).__init__()
        self.kernel_size = kernel_size
        self.out_channels = out_channels
        self.stride = stride
        self.padding = padding
        
        # Kernel generator network: input is a global pooled feature
        self.kernel_gen = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            nn.Linear(in_channels, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, out_channels * in_channels * kernel_size * kernel_size)
        )

        self.in_channels = in_channels
        
    def forward(self, x):
        batch_size, in_c, H, W = x.size()

        # Generate weights dynamically based on input features
        weights = self.kernel_gen(x)  # shape: (B, out_c * in_c * k * k)
        weights = weights.view(batch_size * self.out_channels, self.in_channels, self.kernel_size, self.kernel_size)

        # Reshape input for grouped conv
        x = x.view(1, batch_size * in_c, H, W)
        # out = F.conv2d(x, weight=weights, bias=None, stride=1, padding=self.kernel_size // 2, groups=batch_size)
        out = F.conv2d(
            x, weight=weights, bias=None,
            stride=self.stride, padding=self.padding, groups=batch_size
        )
        # out = out.view(batch_size, self.out_channels, H, W)
        _, _, H_out, W_out = out.shape
        out = out.view(batch_size, self.out_channels, H_out, W_out)
        return out

class ResNet34_Dynamic(nn.Module):
    def __init__(self, input_channels=3, num_classes=100):
        super().__init__()
        self.in_channels = 64
        self.conv1 = DynamicConv(input_channels, 64, kernel_size=7, stride=2, padding=3)
        self.bn1 = nn.BatchNorm2d(64)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(3, 2, 1)
        self.layer1 = self._make_layer(64, 3)
        self.layer2 = self._make_layer(128, 4, 2)
        self.layer3 = self._make_layer(256, 6, 2)
        self.layer4 = self._make_layer(512, 3, 2)
        self.avgpool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Linear(512, num_classes)

    def _make_layer(self, out_channels, blocks, stride=1):
        downsample = None
        if stride != 1 or self.in_channels != out_channels:
            downsample = nn.Sequential(
                nn.Conv2d(self.in_channels, out_channels, 1, stride, bias=False),
                nn.BatchNorm2d(out_channels),
            )
        layers = [BasicBlock(self.in_channels, out_channels, stride, downsample)]
        self.in_channels = out_channels
        for _ in range(1, blocks):
            layers.append(BasicBlock(self.in_channels, out_channels))
        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.relu(self.bn1(self.conv1(x)))
        x = self.maxpool(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.fc(x)
        return x