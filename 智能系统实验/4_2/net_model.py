import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import matplotlib.pyplot as plt
import torchvision


class ICNET(nn.Module):
    def __init__(self):
        super(ICNET, self).__init__()
        self.conv1 = nn.Sequential(
            nn.Conv2d(3, 8, 3, 1, 1),
            nn.ReLU()
        )
        self.conv2 = nn.Sequential(
            nn.Conv2d(8, 8, 3, 1, 1),
            nn.ReLU(),
            nn.MaxPool2d(2, 2)
        )
        self.fc = nn.Sequential(
            nn.Linear(64 * 64 * 8, 1000),
            nn.ReLU(),
            nn.Linear(1000, 1000),
            nn.ReLU(),
            nn.Linear(1000, 2),
            nn.Softmax(dim=1)
        )

    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        x = x.view(x.size()[0], -1)
        x = self.fc(x)
        # output = F.log_softmax(x, dim=1)
        return x


class RESNET(nn.Module):
    def __init__(self):
        super(RESNET, self).__init__()
        self.model = torchvision.models.resnet34(pretrained=True)
        self.model.fc = nn.Linear(512, 2)

    def forward(self, x):
        return self.model(x)


class Net1(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(3, 32, 20, 4)
        self.norm1 = nn.BatchNorm2d(32)
        # nn.init.xavier_uniform_(self.conv1.weight)
        # MaxPool的移动步长默认为kernel_size
        self.maxpool1 = nn.MaxPool2d(kernel_size=2)  # n*32*28*28-->n*32*14*14

        self.conv2 = nn.Conv2d(32, 64, 3, 1, 2)  # n*32*14*14-->n*64*16*16
        self.norm2 = nn.BatchNorm2d(64)
        # nn.init.xavier_uniform_(self.conv2.weight)
        self.maxpool2 = nn.MaxPool2d(2, 2)  # n*64*16*16-->n*64*8*8
        self.dropout1 = nn.Dropout(0.25)
        self.dropout2 = nn.Dropout(0.5)

        self.fc1 = nn.Linear(4096, 128)  # n*4096-->n*4096
        self.fc2 = nn.Linear(128, 2)  # n*4096-->n*10

    def forward(self, x):
        out = F.relu(self.conv1(x))
        out = self.norm1(out)
        out = self.maxpool1(out)
        out = self.dropout1(out)

        out = F.relu(self.conv2(out))
        out = self.norm2(out)
        out = self.maxpool2(out)
        out = self.dropout2(out)

        out = out.view(out.size(0), -1)
        out = F.relu(self.fc1(out))
        out = self.fc2(out)
        return out


class Residual(nn.Module):
    def __init__(self, in_channels, out_channels, use_1x1conv=False, stride=1):
        super(Residual, self).__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1, stride=stride)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1)
        if use_1x1conv:
            self.conv3 = nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=stride)
        else:
            self.conv3 = None
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.bn2 = nn.BatchNorm2d(out_channels)

    def forward(self, X):
        Y = F.relu(self.bn1(self.conv1(X)))
        Y = self.bn2(self.conv2(Y))
        if self.conv3:
            X = self.conv3(X)
        return F.relu(Y + X)


def resnet_block(in_channels, out_channels, num_residuals, first_block=False):
    if first_block:
        assert in_channels == out_channels  # 第一个模块的通道数同输入通道数一致
    blk = []
    for i in range(num_residuals):
        if i == 0 and not first_block:
            blk.append(Residual(in_channels, out_channels, use_1x1conv=True, stride=2))
        else:
            blk.append(Residual(out_channels, out_channels))
    return nn.Sequential(*blk)


class my_resnet(nn.Module):
    def __init__(self):
        super(my_resnet, self).__init__()
        self.net = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=(7, 7), stride=(1, 1), padding=(3, 3)),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),

            resnet_block(32, 32, 2, True),
            resnet_block(32, 64, 2)
        )

        self.fc = nn.Sequential(
            nn.Linear(64, 2),
            nn.Softmax(dim=1)
        )

    def forward(self, x):
        x = self.net(x)
        # x = F.adaptive_avg_pool2d(x, (1, 1))
        x = F.avg_pool2d(x, kernel_size=x.size()[2:])
        x = x.view(x.size(0), -1)
        x = self.fc(x)
        return x


def conv_block(in_channels, out_channels):
    blk = nn.Sequential(nn.BatchNorm2d(in_channels),
                        nn.ReLU(),
                        nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1))
    return blk


class DenseBlock(nn.Module):
    def __init__(self, num_convs, in_channels, out_channels):
        super(DenseBlock, self).__init__()
        net = []
        for i in range(num_convs):
            in_c = in_channels + i * out_channels
            net.append(conv_block(in_c, out_channels))
        self.net = nn.ModuleList(net)
        self.out_channels = in_channels + num_convs * out_channels  # 计算输出通道数

    def forward(self, X):
        for blk in self.net:
            Y = blk(X)
            X = torch.cat((X, Y), dim=1)  # 在通道维上将输入和输出连结
        return X


def transition_block(in_channels, out_channels):
    blk = nn.Sequential(
        nn.BatchNorm2d(in_channels),
        nn.ReLU(),
        nn.Conv2d(in_channels, out_channels, kernel_size=1),
        nn.AvgPool2d(kernel_size=2, stride=2))
    return blk


class densenet(nn.Module):
    def __init__(self):
        super(densenet, self).__init__()
        self.net = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=(7, 7), stride=(1, 1), padding=(3, 3)),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),

            DenseBlock(2, 64, 64),
            transition_block(192, 96),
            DenseBlock(2, 96, 64),
            transition_block(224, 112),

        )

        self.fc = nn.Sequential(
            nn.Linear(112, 2),
            nn.Softmax(dim=1)
        )

    def forward(self, x):
        x = self.net(x)
        # x = F.adaptive_avg_pool2d(x, (1, 1))
        x = F.avg_pool2d(x, kernel_size=x.size()[2:])
        x = x.view(x.size(0), -1)
        x = self.fc(x)
        return x


net = densenet()
x = torch.rand((1, 3, 128, 128))

X = net(x)
print(X.shape)
print(net)

#  TODO 干啥