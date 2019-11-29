import torch
import torch.nn as nn
import torch.nn.functional as F
import math


class Transition(nn.Module):
    def __init__(self, in_planes, out_planes):
        super(Transition, self).__init__()
        self.conv = nn.Conv2d(in_planes, out_planes, kernel_size=1, bias=False)
        self.bn = nn.BatchNorm2d(in_planes)

    def forward(self, x):
        out = self.bn(x)
        out = F.relu(out)
        out = self.conv(out)
        out = F.avg_pool2d(out, 2)
        return out


class BottleNeck(nn.Module):
    def __init__(self, in_planes, growth_rate):
        super(BottleNeck, self).__init__()
        self.bn1 = nn.BatchNorm2d(in_planes)
        self.conv1 = nn.Conv2d(in_planes, 4 * growth_rate, kernel_size=1, bias=False)
        self.bn2 = nn.BatchNorm2d(4 * growth_rate)
        self.conv2 = nn.Conv2d(4 * growth_rate, growth_rate, kernel_size=3, padding=1, bias=False)

    def forward(self, x):
        out = self.conv1(F.relu(self.bn1(x)))
        out = self.conv2(F.relu(self.bn2(out)))
        out = torch.cat([out, x], 1)
        return out


class NormalBottle(nn.Module):
    def __init__(self, in_planes, growth_rate):
        super(NormalBottle, self).__init__()
        self.bn = nn.BatchNorm2d(in_planes)
        self.conv = nn.Conv2d(in_planes, growth_rate, kernel_size=3, padding=1, bias=False)

    def forward(self, x):
        out = F.relu((self.bn(x)))
        out = self.conv(out)
        out = torch.cat([out, x], 1)
        return out


class DenseNet(nn.Module):
    def __init__(self, growth_rate, block, nblocks, reduction=0.5):
        super(DenseNet, self).__init__()
        self.growth_rate = growth_rate

        num_planes = 2 * growth_rate  # 论文中第一个卷积层的output是2*growth rate
        self.conv1 = nn.Conv2d(3, num_planes, kernel_size=3, padding=1, bias=False)

        self.dense1 = self._make_dense_layer(block, num_planes, nblocks[0])
        num_planes += nblocks[0] * growth_rate
        out_planes = int(math.floor(num_planes * reduction))
        self.trans1 = Transition(num_planes, out_planes)
        num_planes = out_planes

        self.dense2 = self._make_dense_layer(block, num_planes, nblocks[1])
        num_planes += nblocks[1] * growth_rate
        out_planes = int(math.floor(num_planes * reduction))
        self.trans2 = Transition(num_planes, out_planes)
        num_planes = out_planes

        self.dense3 = self._make_dense_layer(block, num_planes, nblocks[2])
        num_planes += nblocks[2] * growth_rate

        self.bn = nn.BatchNorm2d(num_planes)
        self.fc = nn.Linear(num_planes, 10)

        # 初始化
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.constant_(m.bias, 0)

    def _make_dense_layer(self, block, in_planes, num):
        layers = []
        for i in range(num):
            layers.append(block(in_planes, self.growth_rate))
            in_planes += self.growth_rate  # densenet每一个bottleneck的输出都是growth rate
        return nn.Sequential(*layers)

    def forward(self, x):
        out = self.conv1(x)
        out = self.trans1(self.dense1(out))
        out = self.trans2(self.dense2(out))
        out = self.dense3(out)
        out = self.bn(out)
        out = F.avg_pool2d(out, 8)
        out = torch.flatten(out, 1)
        out = self.fc(out)
        return out


def main():
    net = DenseNet(12, BottleNeck, [6, 6, 6], reduction=0.5)
    x = torch.randn(3, 3, 32, 32)
    y = net(x)
    print(y.size())


if __name__ == '__main__':
    main()
