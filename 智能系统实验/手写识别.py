import numpy as np
import torch
import torchvision
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt
import torchvision.transforms as transforms


class LeNet(nn.Module):
    def __init__(self):
        super(LeNet, self).__init__()
        self.conv1 = nn.Sequential(
            nn.Conv2d(1, 6, 5, 1, 2),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2)
        )
        self.conv2 = nn.Sequential(
            nn.Conv2d(6, 16, 5),
            nn.ReLU(),
            nn.MaxPool2d(2, 2)
        )
        self.linear = nn.Sequential(
            nn.Linear(16 * 5 * 5, 120),
            nn.Linear(120, 84),
            nn.Linear(84, 10)
        )

    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        x = x.view(x.size()[0], -1)
        x = self.linear(x)
        return x


def train():
    model.train()
    loss = 0.0
    for batch_idx, (data, target) in enumerate(trainloader):
        data, target = data.to(device), target.to(device)
        optimizer.zero_grad()
        output = model(data)
        loss = criterion(output, target)
        loss.backward()
        optimizer.step()
        if batch_idx % log_interval == 0:
            print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                epoch, batch_idx * len(data), len(trainloader.dataset),
                       100. * batch_idx / len(trainloader), loss.item()))


def treasure():
    model.eval()
    test_loss = 0
    correct = 0
    with torch.no_grad():
        for data, target in testloader:
            data, target = data.to(device), target.to(device)
            output = model(data)
            test_loss += criterion(output, target).item()
            pred = output.argmax(dim=1, keepdim=True)
            correct += pred.eq(target.view_as(pred)).sum().item()

    test_loss /= len(testloader.dataset)
    print('\nTest set: Average loss: {:.4f}, Accuracy: {}/{} ({:.2f}%)\n'.format(
        test_loss, correct, len(testloader.dataset),
        100. * correct / len(testloader.dataset)))
    return 100. * correct / len(testloader.dataset), model.state_dict()


transform = transforms.Compose(
    [transforms.ToTensor(),
     transforms.Normalize((0.1307,), (0.3081,))])

trainset = torchvision.datasets.MNIST('./data',
                                      download=True,
                                      train=True,
                                      transform=transform)
testset = torchvision.datasets.MNIST('./data',
                                     download=True,
                                     train=False,
                                     transform=transform)

trainloader = torch.utils.data.DataLoader(trainset, batch_size=64,
                                          shuffle=True, num_workers=0)

testloader = torch.utils.data.DataLoader(testset, batch_size=64,
                                         shuffle=False, num_workers=0)

log_interval = 300
epochs = 5

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
torch.manual_seed(1)
criterion = nn.CrossEntropyLoss()
model = LeNet().to(device)
# optimizer=torch.optim.SGD(model.parameters(),lr=lr,momentum=momentum)
optimizer = optim.Adadelta(model.parameters(), lr=1.0)

if __name__ == '__main__':
    max_acc = float('0')
    for epoch in range(1, epochs + 1):
        train()
        accuracy, net_dict = treasure()
        if accuracy > max_acc:
            max_acc = accuracy
            torch.save({'epoch_record': epoch, 'model': net_dict},
                       f'./手写识别模型_{max_acc}%.pth')
