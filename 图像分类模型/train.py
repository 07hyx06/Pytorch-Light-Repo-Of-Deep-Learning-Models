import torch
import torch.optim as optim
import datetime


def train(device, net, criterion, lr, train_loader, epochs, save_path, print_freq=30):
    optimizer = optim.SGD(net.parameters(), lr=lr, momentum=0.9, weight_decay=1e-4)
    start_time = datetime.datetime.now()

    for epoch in range(1, 1 + epochs):
        iter = 0
        for img, label in train_loader:
            iter += 1

            img = img.to(device)
            label = label.to(device)
            optimizer.zero_grad()

            predict_label = net(img)
            loss = criterion(predict_label, label)

            loss.backward()
            optimizer.step()
            if (iter % print_freq) == 0:
                with torch.no_grad():
                    end_time = datetime.datetime.now()
                    print('epoch:' + str(epoch) + ' | loss=' + str(loss.item()) +
                          ' | time=' + str(end_time - start_time))

        torch.save(net.state_dict(), save_path)
