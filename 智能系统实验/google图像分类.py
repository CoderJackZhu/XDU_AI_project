import torch
import torchvision
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import transforms
from torchvision import datasets
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt

# classes = ['airplane', 'automobile', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck']

transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
])

trainset = datasets.CIFAR10('./data/CIFAR10/cifar-10-python',
                            download=True,
                            train=True,
                            transform=transform)
testset = datasets.CIFAR10('./data/CIFAR10/cifar-10-python',
                           download=True,
                           train=False,
                           transform=transform)

trainloader = DataLoader(trainset, batch_size=64,
                         shuffle=True, num_workers=0)

testloader = DataLoader(testset, batch_size=64,
                        shuffle=False, num_workers=0)


class InceptionA(nn.Module):
    def __init__(self, in_channels):
        super(InceptionA, self).__init__()
        self.branch3x3_1 = nn.Conv2d(in_channels, 16, kernel_size=1)
        self.branch3x3_2 = nn.Conv2d(16, 24, kernel_size=3, padding=1)
        self.branch3x3_3 = nn.Conv2d(24, 24, kernel_size=3, padding=1)

        self.branch5x5_1 = nn.Conv2d(in_channels, 16, kernel_size=1)
        self.branch5x5_2 = nn.Conv2d(16, 24, kernel_size=5, padding=2)

        self.branch1x1 = nn.Conv2d(in_channels, 16, kernel_size=1)

        self.branch_pool = nn.Conv2d(in_channels, 24, kernel_size=1)

    def forward(self, x):
        branch3x3 = self.branch3x3_1(x)
        branch3x3 = self.branch3x3_2(branch3x3)
        branch3x3 = self.branch3x3_3(branch3x3)

        branch5x5 = self.branch5x5_1(x)
        branch5x5 = self.branch5x5_2(branch5x5)

        branch1x1 = self.branch1x1(x)

        branch_pool = F.avg_pool2d(x, kernel_size=3, stride=1, padding=1)
        branch_pool = self.branch_pool(branch_pool)

        outputs = [branch1x1, branch5x5, branch3x3, branch_pool]
        return torch.cat(outputs, dim=1)


class Net(torch.nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.conv1 = torch.nn.Conv2d(3, 10, kernel_size=5)
        self.conv2 = torch.nn.Conv2d(88, 20, kernel_size=5)

        self.incep1 = InceptionA(in_channels=10)
        self.incep2 = InceptionA(in_channels=20)

        self.mp = torch.nn.MaxPool2d(2)
        self.fc = torch.nn.Linear(2200, 10)

    def forward(self, x):
        x = F.relu(self.mp(self.conv1(x)))
        x = self.incep1(x)
        x = F.relu(self.mp(self.conv2(x)))
        x = self.incep2(x)
        x = x.view(x.size()[0], -1)
        x = self.fc(x)
        return x


def train():
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


def measure():
    model.eval()
    test_loss = 0
    correct = 0
    with torch.no_grad():
        for data, target in testloader:
            data = data.to(device)
            target = target.to(device)
            output = model(data)
            test_loss += criterion(output, target).item()
            pred = output.argmax(dim=1, keepdim=True)
            correct += pred.eq(target.view_as(pred)).sum().item()

    test_loss /= len(testloader.dataset)

    print('\nTest set: Average loss: {:.4f}, Accuracy: {}/{} ({:.2f}%)\n'.format(
        test_loss, correct, len(testloader.dataset),
        100. * correct / len(testloader.dataset)))
    return test_loss, 100. * correct / len(testloader.dataset), model.state_dict()


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = Net().to(device)
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adadelta(model.parameters(), lr=1.0)
# optimizer=optim.SGD(model.parameters(),lr=0.01,momentum=0.5)
scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode="max",
                                                       factor=0.45, patience=3)

log_interval = 300
epochs = 20

epoch_list, ls_list, accuracy_list = [], [], []
if __name__ == '__main__':
    max_acc = float('0')
    for epoch in range(1, epochs + 1):
        train()
        ls, accuracy, net_dict = measure()
        scheduler.step(accuracy)

        epoch_list.append(epoch)
        ls_list.append(ls)
        accuracy_list.append(accuracy)

        if accuracy > max_acc:
            max_acc = accuracy
            torch.save({'epoch_record': epoch, 'model': net_dict},
                       f'./google分类模型_{max_acc}%.pth')

    fig = plt.figure(figsize=(20, 8))
    plt.subplot(121)
    plt.plot(epoch_list, ls_list, linestyle=':')
    plt.xlabel('epoch')
    plt.ylabel('loss')
    plt.subplot(122)
    plt.plot(epoch_list, accuracy_list, linestyle=':')
    plt.xlabel('epoch ')
    plt.ylabel('accuracy')
    plt.savefig('./google.png')
    plt.show()
