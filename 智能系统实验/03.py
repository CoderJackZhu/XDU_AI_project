'''
Task: Dara Classification
'''

import numpy as np
import torch
import torch.nn as nn
import matplotlib.pyplot as plt


"""
Stage1: 构建数据集
"""
data = torch.ones(100, 2)
x0 = torch.normal(2*data, 1)
x1 = torch.normal(-2*data, 1)
x = torch.cat((x0,x1),0).type(torch.FloatTensor).cuda()
y0 = torch.zeros(100)
y1 = torch.ones(100)
y = torch.cat((y0,y1)).type(torch.LongTensor).cuda()

"""
Stage2：定义网络
"""
class Net(nn.Module):
    def __init__(self):
        super(Net,self).__init__()
        self.classify = nn.Sequential(
            nn.Linear(2,15),
            nn.ReLU(),
            nn.Linear(15,2),
            nn.Softmax(dim=1)
        )
    def forward(self, x):
        classification = self.classify(x)
        return classification

"""
Stage3：训练模型
"""
net = Net().cuda()
optimizer = torch.optim.SGD(net.parameters(), lr=0.03)
loss_func = nn.CrossEntropyLoss()

plt.figure(figsize=(8,6),dpi=80)
plt.ion()
for epoch in range(100):
    out = net(x)
    loss = loss_func(out, y)
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    classification = torch.max(out,1)[1]
    class_y = classification.cpu().data.numpy()
    target_y = y.cpu().data.numpy()
    accuracy = sum(class_y==target_y)/200

    plt.cla()
    print(f'Epoch={epoch+1}, Accuracy={accuracy}')
    plt.scatter(x.cpu().data.numpy()[:,0],x.cpu().data.numpy()[:,1],c=class_y,s=100,cmap='RdYlGn')
    plt.text(-0.5,-4.5,f'Epoch={epoch+1}, Accuracy={accuracy}',fontdict={'size':18,'color':'red'})
    plt.pause(0.01)

    if epoch == 99:
        plt.savefig('Classification.png')

plt.ioff()
plt.show()


"""
Stage4：测试模型
"""
sample = torch.tensor([[-4,0],[-3,-0.5],[1.5,1.5],[3,1]]).cuda()
print("Sample:", sample)
result = net(sample)
print("Classification result:", torch.max(result,1)[1].data)
