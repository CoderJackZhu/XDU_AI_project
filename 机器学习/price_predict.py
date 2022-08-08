#!/usr/bin/env python
# coding: utf-8


import sklearn
from sklearn import metrics
from sklearn.model_selection import train_test_split
import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torch.utils.data import TensorDataset
import matplotlib.pyplot as plt
import seaborn as sns
from skimage.metrics import mean_squared_error
from sklearn.metrics import r2_score
from sklearn.linear_model import LinearRegression

column_names = ['CRIM', 'ZN', 'INDUS', 'CHAS', 'NOX', 'RM', 'AGE', 'DIS', 'RAD', 'TAX', 'PTRATIO', 'B', 'LSTAT',
                'PRICE']
all_data = pd.read_csv('./housing.csv', header=None, delimiter=r"\s+", names=column_names)

all_data.hist()
plt.show()

all_data.describe()

plt.figure(figsize=(20, 10))
plt.boxplot(all_data)
plt.show()

corr = all_data.corr()

plt.figure(figsize=(20, 10))
sns.heatmap(corr, annot=True, cmap='twilight_r')

data = all_data.iloc[:, :-1]
label = all_data.iloc[:, -1]

data = np.array(data, dtype=float)
label = np.array(label, dtype=float)

for i in range(13):
    plt.figure(figsize=(10, 7))
    plt.grid()
    plt.scatter(data[:, i], label, s=5)  # 横纵坐标和点的大小
    plt.title(column_names[i])
plt.show()

unsF = []  # 次要特征下标
for i in range(data.shape[1]):
    if column_names[i] == 'CHAS':
        unsF.append(i)
data = np.delete(data, unsF, axis=1)  # 删除次要特征

unsT = []  # 房价异常值下标
for i in range(data.shape[1]):
    if label[i] > 46:
        unsT.append(i)
data = np.delete(data, unsT, axis=0)  # 删除样本异常值数据
label = np.delete(label, unsT, axis=0)  # 删除异常房价

data = torch.tensor(data, dtype=torch.float)
label = torch.tensor(label, dtype=torch.float)

X_train, X_test, y_train, y_test = train_test_split(data, label, test_size=0.3, random_state=4)


def train(model, device, train_loader, optimizer, epoch, criterion):
    model.train()
    loss = 0.0
    for i, (data, target) in enumerate(train_loader):
        data, target = data.to(device), target.to(device)
        optimizer.zero_grad()
        output = model(data)
        loss = criterion(output, target.view_as(output))
        loss.backward()
        optimizer.step()
        if i % 100 == 0:
            print('Train Epoch: {} Loss: {:.6f}'.format(
                epoch, loss.item() / len(train_loader)))


def test(model, device, test_loader, criterion):
    model.eval()
    test_loss = 0
    with torch.no_grad():
        for data, target in test_loader:
            data, target = data.to(device), target.to(device)
            output = model(data)

            test_loss += criterion(output, target.view_as(output)).item()  # sum up batch loss

    test_loss /= len(test_loader.dataset)

    print('Test set: Average loss: {:.4f}\n'.format(
        test_loss))
    return test_loss


class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.fc1 = nn.Linear(12, 128)
        self.fc2 = nn.Linear(128, 1)

    def forward(self, x):
        x = self.fc1(x)
        x = F.relu(x)
        x = self.fc2(x)
        return x


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = Net().to(device)
optimizer = torch.optim.Adam(params=model.parameters())
criterion = nn.MSELoss()

trainset = TensorDataset(X_train, y_train)
trainloader = DataLoader(trainset, batch_size=64, shuffle=True, num_workers=0)
testset = TensorDataset(X_test, y_test)
testloader = DataLoader(testset, batch_size=64, shuffle=False, num_workers=0)

epoch_list, loss_list = [], []

for epoch in range(1, 1000):
    train(model, device, trainloader, optimizer, epoch, criterion)
    test_loss = test(model, device, testloader, criterion)
    epoch_list.append(epoch)
    loss_list.append(test_loss)

fig = plt.figure(figsize=(20, 10))
plt.plot(epoch_list, loss_list)
plt.xlabel('epoch')
plt.ylabel('loss')
plt.title('error')
plt.show()


def read(test_loader):
    model.eval()
    output_list, target_list = [], []
    with torch.no_grad():
        for data, target in test_loader:
            model.to('cpu')
            output = model(data).detach().cpu().numpy()
            output_list.extend(output)
            target_list.extend(target.cpu().numpy())
    p = pd.DataFrame(output_list, columns=['predict'])
    p['real'] = target_list
    print(p.head())
    return p


p = read(testloader)

error1 = mean_squared_error(p.iloc[:, 1], p.iloc[:, 0]).round(5)  # 平方差
score1 = r2_score(p.iloc[:, 1], p.iloc[:, 0]).round(5)  # 相关系数

plt.rcParams['font.family'] = "sans-serif"
plt.rcParams['font.sans-serif'] = "SimHei"
plt.rcParams['axes.unicode_minus'] = False
fig1 = plt.figure(figsize=(20, 10))
plt.plot(range(p.shape[0]), p.iloc[:, 1], color='red', linewidth=1, linestyle='-')
plt.plot(range(p.shape[0]), p.iloc[:, 0], color='blue', linewidth=1, linestyle='dashdot')
plt.legend(['真实值', '预测值'])
plt.title('神经网络预测值与准确率对比图')
error1 = "标准差d=" + str(error1) + "\n" + "相关指数R^2=" + str(score1)
plt.xlabel(error1, size=18, color="green")
plt.grid()
plt.show()

lf = LinearRegression()
lf.fit(X_train, y_train)  # 训练数据,学习模型参数
y_predict = lf.predict(X_test)

error2 = mean_squared_error(y_test.numpy(), y_predict).round(5)  # 平方差
score2 = r2_score(y_test, y_predict).round(5)

fig2 = plt.figure(figsize=(20, 10))
plt.plot(range(y_test.shape[0]), y_test, color='red', linewidth=1, linestyle='-')
plt.plot(range(y_test.shape[0]), y_predict, color='blue', linewidth=1, linestyle='dashdot')
plt.legend(['真实值', '预测值'])
plt.title('线性模型预测值与准确率对比图')
error2 = "标准差d=" + str(error2) + "\n" + "相关指数R^2=" + str(score2)
plt.xlabel(error2, size=18, color="green")
plt.grid()
plt.show()
