# -*- coding:utf-8 -*-
# Author : JackZhu

# Data : 2021/5/13 15:25
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import pandas as pd
import numpy as np


class Classifier(nn.Module):
    def __init__(self,in_channel,out_channel):
        super().__init__()
        self.model = nn.Sequential(
            nn.Linear(in_channel, out_channel),
        )
        self.optimizer = torch.optim.SGD(self.parameters(),lr=0.01)
        self.criterion = nn.MSELoss()
        self.counter = 0
        self.progress = []

    def forward(self, inputs):
        y=self.model(inputs)
        y= y.squeeze(-1)
        return y

    def train_model(self, inputs, targets,device):
        inputs,targets=inputs.to(device),targets.to(device)
        outputs = self.forward(inputs)
        loss = self.criterion(outputs, targets)
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        self.counter += 1
        if (self.counter % 10 == 0):
            self.progress.append(loss.item())
        if (self.counter % 10000 == 0):
            print('counter=', self.counter)

    def plot_progress(self):
        df = pd.DataFrame(self.progress, columns=['loss'])
        print(df)
        df.plot(figsize=(16, 8), grid=True)



def get_training_dataset(path):
    file=pd.read_csv(path,header=None)
    data = file.iloc[:, :-1]
    target = file.iloc[:, -1]
    target = pd.get_dummies(target).iloc[:, -1]
    data = np.array(data)
    target = np.array(target)
    data=torch.from_numpy(data).to(torch.float32)
    target=torch.from_numpy(target).to(torch.float32)

    return data,target

if __name__ == '__main__':
    path = './sonar.csv'
    data,target=get_training_dataset(path)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    perceptron=Classifier(data.shape[1],1).to(device)
    for epoch in range(100):
        for i in range(data.shape[0]):
            perceptron.train_model(data[i],target[i],device)
    perceptron.plot_progress()
    acc=0
    for i in range(data.shape[0]):
        input=data[i].to(device)
        result=perceptron.forward(input).cpu().detach()
        if result>0:
            answer=1
        else:
            answer=0
        acc+=(target[i].numpy()==answer)
    print(acc/len(target))
