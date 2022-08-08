# -*- coding:utf-8 -*-
# Author : JackZhu

# Data : 2021/5/13 12:56

import numpy as np
import pandas as pd
import sklearn
import random
from sklearn import datasets
from sklearn.preprocessing import scale
from sklearn.manifold import TSNE
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
from PIL import Image
import cv2


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
    def fit(self,counts=15):
        count=0
        while(count<counts):
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

        return self.centers, self.result

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

    def plot_img(self,row,col):
        img=self.labels.reshape(row,col)
        im = Image.new("RGB", (row, col))  # 创建图片
        for i in range(row):
            for j in range(col):
                if img[i, j] == 0:
                    im.putpixel((i, j), (255, 0, 0))
                if img[i, j] == 1:
                    im.putpixel((i, j), (0, 255, 0))
                if img[i, j] == 2:
                    im.putpixel((i, j), (0, 0, 255))
        im.show()
        im.save('result.jpg')


path='./2.bmp'
# path=f'H:/Python_code/Pattern Recognition/kmeans/kmeans图片/3.bmp'
file=Image.open(path,'r')
file=np.array(file)
row,col,_=file.shape
data=file.reshape(-1,3)
kmeans=Kmeans(data,3)
centers,results=kmeans.fit(10)
kmeans.imshow()
kmeans.plot_img(row,col)
print(centers)
print(results)