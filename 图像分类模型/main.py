from test import test
from train import train
from Dataset import cifar10
import torch
import torch.nn as nn
from ResNet.model import ResNet


if __name__ == '__main__':
    train_loader = cifar10(path='E:/_大二上/_计算机视觉/基础/图像分类', mode='train', batch_size=128, shuffle=True)
    device = torch.device('cuda:0')
    criterion = nn.CrossEntropyLoss().to(device)
    net = ResNet().to(device)
    # net.load_state_dict(torch.load('ResNet/resnet.pkl'))
    train(device, net, criterion, 0.1, train_loader, 2, 'ResNet/resnet.pkl', print_freq=30)
    test_loader = cifar10(path='E:/_大二上/_计算机视觉/基础/图像分类', mode='test', batch_size=512, shuffle=False)
    test(device, net, test_loader)
