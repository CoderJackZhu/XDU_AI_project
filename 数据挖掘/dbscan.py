# -*- coding:utf-8 -*-
# Author : JackZhu

# Data : 2021/5/9 13:55

import sklearn
import scipy.io as scio
import matplotlib.pyplot as plt
import random
import numpy as np

r=0.2
minpts=5

path='./data-密度聚类/2d4c.mat'
data = scio.loadmat(path)['moon']
plt.scatter(data[:,0],data[:,1])
plt.show()
D=data[:,0:2]
row,col=D.shape[0],D.shape[1]
ls=list(range(row))
type_all=[]
noise_ls=[]
j=0
while(ls):
    type_ls=[]
    point=random.choice(ls)
    print(point)
    while(1):
        for i in ls:
            if(np.linalg.norm(D[point]-D[i])<=r):
                type_ls.append(i)
        print(type_ls)
        if(len(type_ls)>=5):
            j=j+1
            ls=type_ls
            continue

        elif(len(type_ls)<5 and len(type_ls)>0):
            pass
        elif(len(type_ls)==0):
            noise_ls.append(point)
            ls.remove(point)
    print(type_ls)
    break
