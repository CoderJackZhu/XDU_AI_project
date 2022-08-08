import torch
import cv2 as cv
import numpy as np
import torch.nn as nn
# from net import ICNET
from torchvision import models
# torch.backends.cudnn.enabled = False

'''
读取数据
'''
x_train = np.load("catdog_train_set.npy")
x_train = torch.tensor(x_train).type(torch.FloatTensor).cuda()
y_train = torch.cat((torch.zeros(70), torch.ones(70)))
y_train = y_train.float().cuda()

'''
构建网络
'''
class ICNET(nn.Module):
    def __init__(self):
        super(ICNET, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(3,8,kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.Conv2d(8,8,kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2, padding=0)
        )
        self.fc = nn.Sequential(
            nn.Linear(64*64*8,1000),
            nn.ReLU(),
            nn.Linear(1000, 1000),
            nn.ReLU(),
            nn.Linear(1000, 2),
            nn.Softmax(dim=1)
        )

    def forward(self, x):
        x = self.conv(x)
        x = x.view(x.size()[0], -1)
        x = self.fc(x)
        return x


model = ICNET().cuda()
# model = resnet_zxy().cuda()
# model = models.resnet34(pretrained=False).cuda()
print('Build the net: Done!')

'''
训练网络
'''
print('Training...')
samplenum = 140
minibatch = 2
w_HR = 128
x0 = np.zeros(minibatch*3*w_HR*w_HR)
x0 = np.reshape(x0,(minibatch,3,w_HR,w_HR))
y0 = np.zeros(minibatch)
x0 = torch.tensor(x0).type(torch.FloatTensor).cuda()
y0 = torch.tensor(y0).type(torch.LongTensor).cuda()
min_loss = float('inf')
loss_func = torch.nn.CrossEntropyLoss()
optimizer = torch.optim.SGD(model.parameters(), lr=0.03)
# optimizer = torch.optim.Adadelta(model.parameters(), lr=0.1)
for epoch in range(10):
    Loss = 0
    for iterations in range(int(samplenum/minibatch)):
        k = 0
        for i in range(iterations*minibatch,iterations*minibatch+minibatch):
            x0[k,0,:,:] = x_train[i,0,:,:]
            x0[k,1,:,:] = x_train[i,1,:,:]
            x0[k,2,:,:] = x_train[i,2,:,:]
            y0[k] = y_train[i]
            k = k+1
        out = model(x0)
        loss = loss_func(out,y0)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        Loss += loss
        print(f'Epoch:{epoch}/10,iteration:{iterations},loss={loss}')
    if Loss < min_loss:
        min_loss = Loss
        torch.save(model, 'best_model.pkl')

'''
测试网络
'''
print('Testing..')
net = torch.load('best_model.pkl')
x_test = np.load("catdog_test_set.npy")/255
x_test = torch.tensor(x_test).type(torch.FloatTensor).cuda()
y_test = torch.cat((torch.zeros(30),torch.ones(30)))
prediction = torch.max(net(x_test),1)[1].cpu().data.numpy()
print(prediction)
label = y_test.data.numpy()
print(f'准确率：{sum(prediction==label)/60}')

