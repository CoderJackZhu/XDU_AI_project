import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torch.nn.functional as F
import matplotlib.pyplot as plt
import cv2
from net_model import ICNET
from net_model import RESNET
from net_model import Net1, my_resnet,densenet
from torch.utils.data import TensorDataset
from torch.utils.data import DataLoader
from torchvision import transforms
import net_model


# from torch.utils.tensorboard import SummaryWriter


def dataloader():
    transform = transforms.Compose([transforms.Resize((128,128)), transforms.ToTensor()])

    trainset = torchvision.datasets.ImageFolder(root='./data/train', transform=transform)
    trainloader = torch.utils.data.DataLoader(trainset, batch_size=16, shuffle=True, num_workers=0)

    testset = torchvision.datasets.ImageFolder(root='./data/test', transform=transform)
    testloader = torch.utils.data.DataLoader(testset, batch_size=8, shuffle=False, num_workers=0)
    return trainloader, testloader


def train(train_loader, model, criterion, optimizer, scheduler, device, epoch):
    model.train()
    train_acc, train_loss = 0.0, 0.0
    for batch_idx, (data, target) in enumerate(train_loader):
        data, target = data.to(device), target.to(device)
        optimizer.zero_grad()
        output = model(data)
        print(target.shape)
        loss = criterion(output, target)
        loss.backward()
        optimizer.step()
        train_loss += loss.item()
        pred = output.argmax(dim=1, keepdim=True)
        train_acc += pred.eq(target.view_as(pred)).sum().item()
    print(
        'Epoch:{}\n Train： Average Loss: {:.6f},Accuracy:{:.2f}%'.format(epoch, train_loss / (batch_idx + 1),
                                                                         100.0 * train_acc / len(train_loader.dataset)))
    scheduler.step(train_acc)


def measure(test_loader, model, criterion, device, epoch):
    model.eval()
    test_loss = 0
    correct = 0
    # writer = SummaryWriter()
    with torch.no_grad():
        for data, target in test_loader:
            data, target = data.to(device), target.to(device)
            output = model(data)
            test_loss += criterion(output, target).item()
            pred = output.argmax(dim=1, keepdim=True)
            correct += pred.eq(target.view_as(pred)).sum().item()
        test_loss /= len(test_loader.dataset)
        print('\nTest: Average loss: {:.4f}, Accuracy: {}/{} ({:.2f}%)\n'.format(
            test_loss, correct, len(test_loader.dataset),
            100. * correct / len(test_loader.dataset)))
    return 100. * correct / len(test_loader.dataset), test_loss, model.state_dict()


def imshow(epoch_list, loss_list, acc_list):
    fig = plt.figure(figsize=(20, 8))
    plt.subplot(121)
    plt.plot(epoch_list, loss_list, linestyle=':')
    plt.xlabel('epoch')
    plt.ylabel('Test loss')
    plt.subplot(122)
    plt.plot(epoch_list, acc_list, linestyle=':')
    plt.xlabel('epoch ')
    plt.ylabel('Test accuracy')
    plt.savefig('./动物分类.png')
    plt.show()


if __name__ == '__main__':
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = my_resnet().to(device)
    # model=ICNET().to(device)
    # model = densenet().to(device)

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adadelta(model.parameters(), lr=1.0)
    # optimizer = optim.Adam(model.parameters(), lr=0.001)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode="max",
                                                           factor=0.85, patience=0)
    train_loader, test_loader = dataloader()
    num_epoch = 50

    max_acc = float('0')
    epoch_list, acc_list, loss_list = [], [], []
    for epoch in range(1, num_epoch + 1):
        train(train_loader, model, criterion, optimizer, scheduler, device, epoch)
        test_acc, test_ls, net_dict = measure(test_loader, model, criterion, device, epoch)
        epoch_list.append(epoch)
        loss_list.append(test_ls)
        acc_list.append(test_acc)
        if test_acc > max_acc:
            max_acc = test_acc
            # torch.save({'epoch_record': epoch, 'model': net_dict},f'./模型/动物分类模型_{max_acc}%.pth')
            torch.save(model, f'./模型/动物分类模型_{max_acc}%.pth')

    imshow(epoch_list, loss_list, acc_list)
