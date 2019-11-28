from torchvision.datasets import CIFAR10
from torchvision import transforms
from torch.utils.data import DataLoader


def cifar10(path, mode, batch_size, shuffle):
    train_data_aug = transforms.Compose([
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.RandomCrop((32, 32), padding=4),
        transforms.ToTensor(),
        transforms.Normalize(mean=(0.4914, 0.4822, 0.4465), std=(0.2023, 0.1994, 0.2010)),
    ])

    test_data_trans = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=(0.4914, 0.4822, 0.4465), std=(0.2023, 0.1994, 0.2010)),
    ])

    if mode == 'train':
        data = CIFAR10(root=path, train=True, transform=train_data_aug)
    else:
        data = CIFAR10(root=path, train=False, transform=test_data_trans)

    return DataLoader(data, batch_size=batch_size, shuffle=shuffle)


if __name__ == '__main__':
    loader = cifar10(path='E:/_大二上/_计算机视觉/基础/图像分类', mode='train', batch_size=1, shuffle=False)
    for img, label in loader:
        print(img.size())
        break

