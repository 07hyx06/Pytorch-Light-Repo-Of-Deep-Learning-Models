import torch


def test(device, net, loader):
    correct = 0
    total = 0
    with torch.no_grad():
        for data, label in loader:
            data = data.to(device)
            output = net(data).to(device)
            pred = torch.argmax(output, 1).to('cpu')
            correct += (pred == label).sum().float().item()
            total += len(label)
    print('accuracy on dataset:'+str(correct / total))
