import numpy as np
import torch
import torch.nn as nn
import matplotlib.pyplot as plt


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

data=torch.ones(100,2)
x0=torch.normal(2*data,1)
x1=torch.normal(-2*data,1)
x=torch.cat((x0,x1),0).type(torch.FloatTensor).to(device)
y0=torch.zeros(100)
y1=torch.ones(100)
y=torch.cat((y0,y1)).type(torch.LongTensor).to(device)

class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.classify=nn.Sequential(
            nn.Linear(2,15),
            nn.ReLU(),
            nn.Linear(15,2),
            nn.Softmax(dim=1)
        )
    def forward(self,x):
        classification=self.classify(x)
        return  classification


net=Net().to(device)
optimizer=torch.optim.SGD(net.parameters(),lr=0.03)
loss_func=nn.CrossEntropyLoss()

epoch_list,loss_list=[],[]
for epoch in range(1000):
    out=net(x)
    loss=loss_func(out,y)
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    epoch_list.append(epoch)
    loss_list.append(loss.cpu().detach())
    if epoch%10==0:
        print(f"epoch{epoch},loss:{loss}")

classification=torch.max(out,1)[1]
class_y=classification.cpu().data.numpy()
target_y=y.cpu().data.numpy()
plt.figure()
plt.scatter(x.cpu().data.numpy()[:,0],x.cpu().data.numpy()[:,1],c=class_y,s=100,cmap='RdYlGn')
accuracy=(class_y==target_y).sum()/200
plt.text(1.5,-4,f'Accuracy={accuracy}',fontdict={'size':20,'color':'red'})
plt.figure()
plt.plot(epoch_list,loss_list,':')
plt.xlabel('epoch')
plt.ylabel('loss')
plt.show()
