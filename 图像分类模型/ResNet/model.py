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
        out += self.extra(x)  # 恒等变换
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

    def forward(self, x):
        x = self.conv1(x)  # [n,3,32,32] -> [n,16,32,32]
        x = self.layer1(x)  # [n,16,32,32] -> [n,16,16,16]
        x = self.layer2(x)  # [n,16,16,16] -> [n,32,8,8]
        x = self.layer3(x)  # [n,32,8,8] -> [n,64,8,8]
        x = self.avg_pool(x)  # [n,64,8,8] -> [n,64,1,1]
        x = torch.flatten(x, 1)  # [n,64,1,1] -> [n,64]
        x = self.fc(x)  # [n,64] -> [n,10]
        return x


def main():
    net = ResNet()
    x = torch.randn(3, 3, 32, 32)
    print(net(x).size())


if __name__ == '__main__':
    main()
