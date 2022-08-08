import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torch.nn.functional as F
import matplotlib.pyplot as plt
import cv2
from net_model import ICNET
from net_model import RESNET
from net_model import animal
from torch.utils.data import TensorDataset
from torch.utils.data import DataLoader

m1, m2 = 70, 30
l_x = 128
train_set = np.zeros(m1 * 2 * l_x * l_x * 3)
train_set = np.reshape(train_set, (m1 * 2, 3, l_x, l_x))

test_set = np.zeros(m2 * 2 * l_x * l_x * 3)
test_set = np.reshape(test_set, (m2 * 2, 3, l_x, l_x))

train_label = np.zeros(m1 * 2)
test_label = np.zeros(m2 * 2)

success_mark = 0
for i in range(m1):
    path1 = f'./sample/cat.{i}.jpg'
    path2 = f'./sample/dog.{i}.jpg'
    img1 = cv2.imread(path1)
    img2 = cv2.imread(path2)
    img1 = cv2.resize(img1, (l_x, l_x))
    img2 = cv2.resize(img2, (l_x, l_x))

    train_set[i, 0, :, :] = img1[:, :, 0]
    train_set[i, 1, :, :] = img1[:, :, 1]
    train_set[i, 2, :, :] = img1[:, :, 2]
    train_label[i] = 0
    success_mark += 1

    train_set[m1 + i, 0, :, :] = img2[:, :, 0]
    train_set[m1 + i, 1, :, :] = img2[:, :, 1]
    train_set[m1 + i, 2, :, :] = img2[:, :, 2]
    train_label[m1 + i] = 1
    success_mark += 1

for i in range(m2):
    path3 = f'./sample/cat.{i + m1}.jpg'
    path4 = f'./sample/dog.{i + m1}.jpg'
    img3 = cv2.imread(path3)
    img4 = cv2.imread(path4)
    img3 = cv2.resize(img3, (l_x, l_x))
    img4 = cv2.resize(img4, (l_x, l_x))

    test_set[i, 0, :, :] = img3[:, :, 0]
    test_set[i, 1, :, :] = img3[:, :, 1]
    test_set[i, 2, :, :] = img4[:, :, 2]
    test_label[i] = 0
    success_mark += 1

    test_set[m2 + i, 0, :, :] = img4[:, :, 0]
    test_set[m2 + i, 1, :, :] = img4[:, :, 1]
    test_set[m2 + i, 2, :, :] = img4[:, :, 2]
    test_label[m2 + i] = 1
    success_mark += 1

if success_mark == 200:
    np.save('train_set.npy', train_set)
    np.save('test_set.npy', test_set)

np.save('test_label.npy',test_label)



device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
#model = ICNET().to(device)
#model=animal().to(device)
model = RESNET().to(device)

criterion = nn.CrossEntropyLoss()
optimizer = optim.Adadelta(model.parameters(), lr=1.0)
#optimizer=optim.SGD(model.parameters(),lr=10e-5,momentum=0.5)

samplenum = 140
minibatch = 2
w_HR = 128
x0 = np.zeros(minibatch * 3 * w_HR * w_HR)
x0 = np.reshape(x0, (minibatch, 3, w_HR, w_HR))
y0 = np.zeros(minibatch)
x0 = torch.tensor(x0).type(torch.FloatTensor).to(device)
y0 = torch.tensor(y0).type(torch.LongTensor).to(device)

train_set = torch.tensor(train_set).type(torch.LongTensor).to(device)
min_loss = float('inf')
for epoch in range(1,10):
    for iterations in range(int(samplenum / minibatch)):
        model.train()
        loss=0.0
        k = 0
        for i in range(iterations * minibatch, min(samplenum, iterations * minibatch + minibatch)):
            x0[k, 0, :, :] = train_set[i, 0, :, :]
            x0[k, 1, :, :] = train_set[i, 1, :, :]
            x0[k, 2, :, :] = train_set[i, 2, :, :]
            y0[k] = train_label[i]
            k = k + 1

        out = model(x0)
        #out_real = torch.max(out, 1)[1]
        loss = criterion(out, y0)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
    print(f"epoch{epoch},loss:{loss.item()}")
    if loss < min_loss:
        min_loss = loss
        torch.save(model,'./动物分类.pkl')

test_feature = torch.tensor(test_set).type(torch.FloatTensor).cuda()
test_label = torch.tensor(test_label).type(torch.FloatTensor).cuda()

test_dataset = TensorDataset(test_feature, test_label)
test_loader = DataLoader(test_dataset, batch_size=4, shuffle=True)


def test_net():
    correct = 0.0
    with torch.no_grad():
        for data, target in test_loader:
            model.eval()
            data, target = data.to(device), target.to(device)
            output = model(data)
            pred = output.argmax(dim=1, keepdim=True)
            correct += pred.eq(target.view_as(pred)).sum().item()
            #pred=torch.max(output,1)[1].cpu().data.numpy()
            #tar=target.cpu().data.numpy()
            #correct+=sum(tar==pred)
    print('accuracy:{:.2f}%'.format(100.0 * correct / len(test_loader.dataset)))


if __name__ == '__main__':
    test_net()