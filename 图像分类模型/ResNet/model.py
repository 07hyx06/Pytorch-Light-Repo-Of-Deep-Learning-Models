import torch
import torch.nn as nn
import torch.nn.functional as F


class ResBlock(nn.Module):
    def __init__(self, in_planes, out_planes, stride):
        super(ResBlock, self).__init__()
        self.layer = nn.Sequential(
            nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride, padding=1, bias=False),
            nn.BatchNorm2d(out_planes),
            nn.ReLU(inplace=True),

            nn.Conv2d(out_planes, out_planes, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(out_planes),
        )
        self.extra = nn.Sequential()
        self.bn = nn.BatchNorm2d(out_planes)
        if stride != 1 or in_planes != out_planes:
            self.extra = nn.Sequential(
                nn.Conv2d(in_planes, out_planes, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(out_planes)
            )

    def forward(self, x):
        out = self.layer(x)
        out += self.extra(x)
        out = F.relu(out)
        return out


class ResNet(nn.Module):
    def __init__(self):
        super(ResNet, self).__init__()
        self.in_planes = 16
        self.conv1 = nn.Sequential(
            nn.Conv2d(3, 16, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(16),
            nn.ReLU(inplace=True),
        )
        self.layer1 = self._make_layer(ResBlock, 16, 3, stride=1)
        self.layer2 = self._make_layer(ResBlock, 32, 3, stride=2)
        self.layer3 = self._make_layer(ResBlock, 64, 3, stride=2)
        self.avg_pool = nn.AvgPool2d(kernel_size=8, stride=1)
        self.fc = nn.Linear(64, 10)

    def _make_layer(self, block, out_planes, num, stride):
        strides = [stride] + [1] * (num - 1)
        layers = []
        for stride in strides:
            layers.append(block(self.in_planes, out_planes, stride))
            self.in_planes = out_planes
        return nn.Sequential(*layers)

    # 前向传播
    def forward(self, x):
        x = self.conv1(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.avg_pool(x)
        x = torch.flatten(x, 1)
        x = self.fc(x)
        return x


def main():
    net = ResNet()
    x = torch.randn(3, 3, 32, 32)
    print(net(x).size())


if __name__ == '__main__':
    main()
