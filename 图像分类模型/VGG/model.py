import torch
import torch.nn as nn
from torch.nn import functional as F


class VGG(nn.Module):
    def __init__(self):
        super(VGG, self).__init__()
        self.layer1 = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 64, kernel_size=3, padding=1),
            nn.MaxPool2d(kernel_size=2, stride=2),
        )
        self.layer2 = nn.Sequential(
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(128, 128, kernel_size=3, padding=1),
            nn.MaxPool2d(kernel_size=2, stride=2),
        )
        self.layer3 = nn.Sequential(
            nn.Conv2d(128, 256, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 256, kernel_size=3, padding=1),
            nn.MaxPool2d(kernel_size=2, stride=2),
        )
        self.classifier = nn.Sequential(
            nn.Linear(256 * 4 * 4, 128),
            nn.BatchNorm1d(128),
            nn.ReLU(inplace=True),
            nn.Linear(128, 10),
        )

    def forward(self, x):
        x = self.layer1(x)  # [n,3,32,32] -> [n,64,16,16]
        x = F.relu(x)
        x = self.layer2(x)  # [n,64,16,16] -> [n,128,8,8]
        x = F.relu(x)
        x = self.layer3(x)  # [n,128,8,8] -> [n,256,4,4]
        x = F.relu(x)
        x = torch.flatten(x, 1)  # [n,256,4,4] -> [n,4096]
        x = self.classifier(x)  # [n,4096] -> [n,10]
        return x


if __name__ == '__main__':
    net = VGG()
    x = torch.randn(3, 3, 32, 32)
    print(net(x).size())
