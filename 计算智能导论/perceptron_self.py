# -*- coding:utf-8 -*-
# Author : JackZhu

# Data : 2021/5/20 18:21
import numpy as np
import pandas as pd
import torch
import matplotlib.pyplot as plt
import scipy.io as scio
from sklearn.manifold import TSNE
from sklearn.decomposition import PCA

class Perceptron():
    def __init__(self,input_num,f):
        self.input_num=input_num
        self.weights=np.ones(input_num)
        self.bias=2.0
        self.activation=f

    def __str__(self):
        return f'weight={self.weights},bias={self.bias}'

    def predict(self,inputs):
        return self.activation(np.dot(inputs,self.weights)+self.bias)

    def train(self,inputs,labels,rate=0.1):
        loss = 0
        for j in range(inputs.shape[0]):
            output = self.predict(inputs[j])
            loss += labels[j] * output
            self.weights = self.weights + rate * (labels[j] - output) * inputs[j]
            self.bias = self.bias + rate * (labels[j] - output)
        loss =-loss/np.linalg.norm(self.weights)
        return loss
def f(x):
    return 1 if x>0 else 0


def get_data():
    path = './sonar.csv'
    file = pd.read_csv(path, header=None)
    data = file.iloc[:, :-1]
    target = file.iloc[:, -1]
    target = pd.get_dummies(target).iloc[:, -1]
    data = np.array(data)
    target = np.array(target)
    return data,target

# def get_data():
#     data=torch.ones(100,2)
#     x0=torch.normal(2*data,1.5)
#     x1=torch.normal(-2*data,1.5)
#     x=torch.cat((x0,x1),0)
#     y0=torch.zeros(100)
#     y1=torch.ones(100)
#     y=torch.cat((y0,y1))
#     data=np.array(x)
#     target=np.array(y)
#     return data,target

# def get_data():
#     path = './long.mat'
#     file = scio.loadmat(path)['long1']
#     data = file[:, 0:2]
#     target = file[:, 2]
#     return data,target

def plot_result(perceptron,result):
    knew = -perceptron.weights[0] / perceptron.weights[1]
    bnew = -perceptron.bias / perceptron.weights[1]
    x = np.linspace(-5, 5)
    y = lambda x: knew * x + bnew
    plt.xlim(-5,5)
    plt.ylim(-1,2)
    plt.plot(x, y(x), 'b--')
    plt.scatter(data[:, 0], data[:, 1], c=result)
    plt.title('Binary Classification')
    plt.show()

def imshow(data,result):
    tsne = TSNE(n_components=2, learning_rate=100).fit_transform(data)
    pca = PCA().fit_transform(data)
    plt.figure(figsize=(12, 6))
    plt.subplot(121)
    plt.scatter(tsne[:, 0], tsne[:, 1], c=result)
    plt.title('t-SNE')
    plt.subplot(122)
    plt.scatter(pca[:, 0], pca[:, 1], c=result)
    plt.title('PCA')
    plt.colorbar()
    plt.show()

def plot_loss(loss_ls,epoch_ls,acc_ls):
    fig = plt.figure(figsize=(20, 8))
    plt.subplot(121)
    plt.plot(epoch_ls,loss_ls)
    plt.title('Loss')
    plt.subplot(122)
    plt.plot(epoch_ls,acc_ls)
    plt.show()

if __name__=='__main__':
    epochs = 50000
    data,target=get_data()
    perceptron=Perceptron(data.shape[1],f)
    print(perceptron)
    loss_ls,epoch_ls,acc_ls=[],[],[]
    for i in range(epochs):
        ls=perceptron.train(data,target)
        if i%100==0:
            loss_ls.append(ls)
            epoch_ls.append(i)
            acc, result = 0,0
            for i in range(data.shape[0]):
                result = perceptron.predict(data[i])
                acc += (result == target[i])/len(target)
            acc_ls.append(acc)
    plot_loss(loss_ls,epoch_ls,acc_ls)
    print(perceptron)
    imshow(data,target)
    print('acc={}'.format(acc_ls[-1]))
