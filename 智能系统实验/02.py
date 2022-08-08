'''
Task: Dara Regression
'''

import numpy as np
import torch
import math
import torch.nn as nn
import matplotlib.pyplot as plt


"""
Stage1: 构建数据集
"""
# 构造等差数列并转为二维数组
x = torch.unsqueeze(torch.linspace(-np.pi,np.pi,100),dim=1)
# 添加随机数
y = torch.sin(x)+0.5*torch.rand(x.size())
x = x.clone().detach().cuda()
y = y.clone().detach().cuda()

"""
Stage2：定义网络
"""
class Net(nn.Module):
    def __init__(self):
        super(Net,self).__init__()
        self.predict = nn.Sequential(
            nn.Linear(1,10),
            nn.ReLU(),
            nn.Linear(10,1)
        )
    def forward(self, x):
        prediciton = self.predict(x)
        return prediciton

"""
Stage3：训练模型
"""
epochNum = 1000
net = Net().cuda()
optimizer = torch.optim.SGD(net.parameters(), lr=0.062)
loss_func = nn.MSELoss()

plt.figure(figsize=(8,6),dpi=80)
plt.ion()
for epoch in range(epochNum):
    out = net(x)
    loss = loss_func(out, y)
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    if epoch % 100 == 0:
        plt.cla()
        plt.scatter(x.cpu().detach().numpy(), y.cpu().detach().numpy())
        plt.plot(x.cpu().detach().numpy(), out.cpu().detach().numpy())
        plt.pause(1)
        print(f"epoch:{epoch},loss:{loss}")
    if epoch == epochNum-1:
        plt.clf()
        plt.scatter(x.cpu().detach().numpy(), y.cpu().detach().numpy())
        plt.plot(x.cpu().detach().numpy(), out.cpu().detach().numpy())
        plt.savefig('RegressionResult.png')
plt.ioff()
plt.show()

"""
Stage4：测试模型
"""
sample = torch.unsqueeze(torch.tensor([-math.pi,-math.pi/2,0,math.pi/2,math.pi]),dim=1)
sample = sample.clone().detach().cuda()
label = torch.sin(sample)
label = label.clone().detach().cuda()
print("Input is：", sample)
result = net(sample)
print("Output is：", result.cpu().detach().numpy)
Loss = loss_func(result, label)
print("Loss is：", Loss)

