from test import test
from train import train
from Dataset import cifar10
import torch
import torch.nn as nn
from ResNet.model import ResNet
from VGG.model import VGG
from DenseNet.model import DenseNet, BottleNeck

if __name__ == '__main__':
    # 数据集地址
    path = 'E:/_大二上/_计算机视觉/基础/图像分类'
    # 模型保存地址
    load_path = 'ResNet/resnet.pkl'
    # 构造训练集loader
    train_loader = cifar10(path=path, mode='train', batch_size=64, shuffle=True)
    # gpu
    device = torch.device('cuda:0')
    # 交叉熵损失
    criterion = nn.CrossEntropyLoss().to(device)
    # net = ResNet().to(device)
    net = DenseNet(12, BottleNeck, [6, 6, 6], reduction=0.5).to(device)
    # net = VGG().to(device)
    # 读取保存的模型参数，继续训练
    # net.load_state_dict(torch.load('ResNet/resnet.pkl'))
    train(device, net, criterion, 0.1, train_loader, 2, load_path, print_freq=30)
    test_loader = cifar10(path=path, mode='test', batch_size=512, shuffle=False)
    test(device, net, test_loader)
