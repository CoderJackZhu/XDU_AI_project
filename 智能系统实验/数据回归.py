import numpy as np
import torch
import torch.nn as nn
import matplotlib.pyplot as plt

# plt.rcParams['font.sans-serif'] = ['SimHei']

x = torch.unsqueeze(torch.linspace(-np.pi, np.pi, 100), dim=1)
y = torch.sin(x) + 0.5 * torch.rand(x.size())


class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.predict = nn.Sequential(
            nn.Linear(1, 10),
            nn.ReLU(),
            nn.Linear(10, 1)
        )

    def forward(self, x):
        prediction = self.predict(x)
        return prediction


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
net = Net().to(device)
x, y = x.to(device), y.to(device)
optimizer = torch.optim.SGD(net.parameters(), lr=0.05)
loss_func = nn.MSELoss()


fig = plt.figure()
plt.ion()
epoch_list, loss_list = [], []
for epoch in range(1, 1001):
    out = net(x)
    loss = loss_func(out, y)
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    plt.scatter(x.cpu(), y.cpu(), color='y')
    epoch_list.append(epoch)
    loss_list.append(loss.cpu().detach())
    if epoch % 100 == 0:
        print(f"epoch:{epoch},loss:{loss}")
plt.plot(x.cpu(), net(x).cpu().detach(), color='r', linestyle="-")
plt.xlabel('x')
plt.ylabel('y')
plt.title('result')
fig = plt.figure()
plt.plot(epoch_list, loss_list)
plt.xlabel('epoch')
plt.ylabel('loss')
plt.title('error')
plt.ioff()
plt.show()
