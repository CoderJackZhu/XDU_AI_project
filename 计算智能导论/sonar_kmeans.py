# -*- coding: utf-8 -*-
"""
Created on Sun May 30 15:28:26 2021

@author: tremble
"""

# -*- coding:utf-8 -*-
# Author : JackZhu

# Data : 2021/5/10 23:10
import numpy as np
import pandas as pd
import sklearn
import random
from sklearn import datasets
from sklearn.preprocessing import scale
from sklearn.manifold import TSNE
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
from sklearn.metrics import silhouette_score


class Kmeans():
    def __init__(self,dat,k):
        data=scale(dat)
        self.data=data
        self.row, self.col = data.shape
        self.k=k
        self.centers=np.ndarray((k,self.col))
        choices=random.choices(range(self.row),k=k)
        for i in range(k):
            self.centers[i,:]=self.data[choices[i],:]
    def fit(self):
        count=0
        while(count<40):
            self.labels=np.zeros((self.row))
            for i in range(self.data.shape[0]):
                dis=[]
                for j in range(self.k):
                    dis.append(np.linalg.norm(self.data[i,:]-self.centers[j,:],axis=0))
                lab=np.argmin(dis,axis=0)
                self.labels[i]=lab
            self.result={}
            for i in range(self.k):
                type=np.where(self.labels==i)[0]
                self.result[i]=type
                if len(type)==0:
                    self.centers[i, :] =0
                else:
                    self.centers[i,:]=np.mean(self.data[type,:],axis=0)
            count+=1
        return self.centers, self.result,self.labels

    def imshow(self):
        tsne = TSNE(n_components=2, learning_rate=100).fit_transform(self.data)
        pca = PCA().fit_transform(self.data)
        plt.figure(figsize=(12, 6))
        plt.subplot(121)
        plt.scatter(tsne[:, 0], tsne[:, 1], c=self.labels)
        plt.title('t-SNE')
        plt.subplot(122)
        plt.scatter(pca[:, 0], pca[:, 1], c=self.labels)
        plt.title('PCA')
        plt.colorbar()
        plt.show()


if __name__ == "__main__":
    path = './sonar.csv'
    file = pd.read_csv(path, header=None)
    data = file.iloc[:, :-1]
    target = file.iloc[:, -1]
    target = pd.get_dummies(target).iloc[:, -1]
    data = np.array(data)
    target = np.array(target)
    kmeans=Kmeans(data,2)
    centers,results,labels=kmeans.fit()
    kmeans.imshow()
    s = silhouette_score(data, labels)
    print(centers)
    print(results)
    print(s)