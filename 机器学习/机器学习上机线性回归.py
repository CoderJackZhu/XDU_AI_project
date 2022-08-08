# -*- coding: utf-8 -*-
"""
Created on Fri Apr  9 13:25:47 2021

@author: tremble
"""

import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt

data = pd.read_csv('./advertising.csv')
features = np.array(data.iloc[:, 0:3])
features = torch.tensor(features, dtype=torch.float)
labels = np.array(data.iloc[:, 3])
labels = torch.tensor(labels, dtype=torch.float)
dataset = torch.utils.data.TensorDataset(features, labels)
dataloader = torch.utils.data.DataLoader(dataset, batch_size=4, shuffle=True)


class Linear_Net(nn.Module):
    def __init__(self, n_features):
        super(Linear_Net, self).__init__()
        self.linear = nn.Linear(n_features, 1)

    def forward(self, x):
        x = self.linear(x)
        return x


net = Linear_Net(3)
torch.nn.init.normal_(net.linear.weight, mean=0, std=0.001)
torch.nn.init.constant_(net.linear.bias, val=0)
loss = nn.MSELoss()
optimizer = optim.Adam(net.parameters(), lr=0.01)
epochs = 50
epoch_list, avg_l_list = [], []
for epoch in range(1, epochs + 1):
    loss_sum = 0
    for x, y in dataloader:
        output = net(x)
        l = loss(y.view(-1, 1), output)
        optimizer.zero_grad()
        l.backward()
        optimizer.step()
        loss_sum += l.detach()
    epoch_list.append(epoch)
    avg_l_list.append(loss_sum / labels.shape[0])
    print('Epoch:{},Loss:{:.2f}'.format(epoch, loss_sum / labels.shape[0]))
plt.plot(epoch_list, avg_l_list)
plt.title('error')
plt.xlabel('epoch')
plt.ylabel('loss')
plt.show()

print(net.linear.weight, net.linear.bias)
