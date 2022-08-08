import torch
from torchvision import  transforms
import torch.nn as nn
import torch.nn.functional as func
import torch.optim as optim
import os
import numpy as np
import pandas as pd
from PIL import Image
from torch.optim.lr_scheduler import StepLR
from torch.utils.data import Dataset
from torchvision.transforms import ToTensor


#加载数据集
class MyDataset(Dataset):

    def __init__(self, path, mode="test", transformer=ToTensor()):
        info_path = f"{path}/{mode}_info.txt"
        path = f"{path}/{mode}"
        info = pd.read_csv(info_path, sep=" ", header=None)
        labels = np.array(info.iloc[:-1, 0])
        img_paths = []
        for i, label in enumerate(labels):
            for file_name in os.listdir(f"{path}/{label}"):
                img_paths.append((i, f"{path}/{label}/{file_name}"))
        self.img_path = img_paths
        self.transformer = transformer

    def __len__(self):
        return len(self.img_path)

    def __getitem__(self, idx):
        i, path = self.img_path[idx]
        image = Image.open(path)
        return self.transformer(image), i

#设置网络结构
class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.conv1 = nn.Conv2d(1, 3, 3, 1, 1)
        self.conv2 = nn.Conv2d(3, 16, 3, 1, 1)
        self.conv3 = nn.Conv2d(16, 32, 3, 1, 1)
        self.conv4 = nn.Conv2d(32, 64, 3, 1, 1)
        self.conv5 = nn.Conv2d(64, 128, 3, 1, 1)
        self.dropout1 = nn.Dropout(0.25)
        self.dropout2 = nn.Dropout(0.25)
        self.dropout3 = nn.Dropout(0.25)
        self.dropout4 = nn.Dropout(0.25)
        self.dropout5 = nn.Dropout(0.25)
        self.fc1 = nn.Linear(128*4*4, 4096)
        self.fc2 = nn.Linear(4096, 4096)
        self.fc3 = nn.Linear(4096, 1024)
        self.fc4 = nn.Linear(1024, 500)

    def forward(self, x):
        x = self.conv1(x)
        x = func.relu(x)
        x = func.max_pool2d(x, 2)  # 64
        x = self.conv2(x)
        x = func.relu(x)
        x = func.max_pool2d(x, 2)  # 32
        x = self.dropout1(x)
        x = self.conv3(x)
        x = func.relu(x)
        x = func.max_pool2d(x, 2)  # 16
        x = self.dropout2(x)
        x = self.conv4(x)
        x = func.relu(x)
        x = func.max_pool2d(x, 2)  # 8
        x = self.dropout3(x)
        x = self.conv5(x)
        x = func.relu(x)
        x = func.max_pool2d(x, 2)  # 4
        x = self.dropout4(x)
        x = torch.flatten(x, 1)
        x = self.fc1(x)
        x = func.relu(x)
        x = self.dropout3(x)
        x = self.fc2(x)
        x = func.relu(x)
        x = self.fc3(x)
        x = func.relu(x)
        x = self.fc4(x)
        output = func.log_softmax(x, dim=1)
        return output


def train(model, device, train_loader, optimizer, epoch, criterion):
    model.train()
    for i, (data, target) in enumerate(train_loader):
        data, target = data.to(device), target.to(device)
        optimizer.zero_grad()
        output = model(data)
        loss = criterion(output, target)
        loss.backward()
        optimizer.step()
        if i % 100 == 0:
            print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                epoch, i * len(data), len(train_loader.dataset),
                       100. * i / len(train_loader), loss.item()))


def test(model, device, test_loader, criterion):
    model.eval()
    test_loss = 0
    correct = 0
    with torch.no_grad():
        for data, target in test_loader:
            data, target = data.to(device), target.to(device)
            output = model(data)
            test_loss += criterion(output, target).item()  # sum up batch loss
            pred = output.argmax(dim=1, keepdim=True)  # get the index of the max log-probability
            correct += pred.eq(target.view_as(pred)).sum().item()

    test_loss /= len(test_loader.dataset)

    print('\nTest set: Average loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)\n'.format(
        test_loss, correct, len(test_loader.dataset),
        100. * correct / len(test_loader.dataset)))
    return correct / len(test_loader.dataset)


#是否可用gpu
use_cuda = torch.cuda.is_available()
#设置随机数种子
if use_cuda:
    torch.cuda.manual_seed(1)
else:
    torch.manual_seed(1)
    
device = torch.device("cuda" if use_cuda else "cpu")
transform = transforms.Compose([
        transforms.Resize((128, 128)),
        transforms.ToTensor(),
        lambda _: _.mean(0).unsqueeze(0),
        transforms.Normalize((0.1307,), (0.3081,)),
    ])

# datasets
train_dataset = MyDataset("./dataset", "train", transformer = transform)
test_dataset = MyDataset("./dataset", "test", transformer = transform)

# dataloaders
trainloader = torch.utils.data.DataLoader(train_dataset, batch_size=4,
                                          shuffle=True, num_workers=0)

testloader = torch.utils.data.DataLoader(test_dataset, batch_size=4,
                                         shuffle=False, num_workers=0)
#交叉熵损失
criterion = nn.CrossEntropyLoss()


net = torch.load(r"D:/RZZ_net7_0.9072.pt")

if use_cuda:
    model = nn.DataParallel(net).cuda()
else:
    model = net.to(device)

#optimizer = optim.SGD(model.parameters(), lr=0.001, momentum=0.9)
optimizer = optim.Adadelta(model.parameters(), lr=1.0)

#设置不断缩小步长
scheduler = StepLR(optimizer, step_size=1, gamma=0.7)
for epoch in range(1, 10):
    train(model, device, trainloader, optimizer, epoch, criterion)
    test_acc = test(model, device, testloader, criterion)
    torch.save(model, f"rzz_net{epoch}_{test_acc:.4f}.pth")  # 每轮保存模型
    scheduler.step()
