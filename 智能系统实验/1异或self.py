import torch
import torch.nn as nn
x=[[0,0],[0,1],[1,0],[1,1]]
y=[[0],[1],[1],[0]]
x=torch.FloatTensor(x).cuda()
y=torch.FloatTensor(y).cuda()
net=nn.Sequential(
    nn.Linear(2,20),
    nn.ReLU(),
    nn.Linear(20,1),
    nn.Sigmoid()
).cuda()

optimizer=torch.optim.SGD(net.parameters(),lr=0.05)
loss_func=nn.MSELoss()
for epoch in range(5000):
    out=net(x)
    loss=loss_func(out,y)
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    if epoch%100==0:
        print(f"迭代次数：{epoch}")
        print(f"误差:{loss}")

out=net(x).cpu()
print(f"out:{out.data}")
torch.save(net,'./异或net.pkl')

test_net=torch.load('异或net.pkl')
test_net.eval()
test_x=[[0,0]]
test_x=torch.FloatTensor(test_x).cuda()
out_final=test_net(test_x)
if out_final>0.5:
    print('1')
else:
    print('0')