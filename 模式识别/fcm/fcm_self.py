# -*- coding: utf-8 -*-
"""
Created on Tue Nov  3 19:28:19 2020

@author: tremble
"""

import pandas as pd
import numpy as np
from sklearn.datasets import load_iris
import matplotlib.pyplot as plt
import random


def eucliDist(A,B):
    return np.sqrt(np.sum(np.square(A - B)))

class fcm(object):
    def __init__(self,data,c,alpha):
        self.data = data
        self.c= c
        row, col = data.shape
        u=np.zeros((row,c))
        for i in range(row):
            for j in range(c):
                u[i,j]=1/c
        self.row = row
        self.col = col
        self.u = u
        self.alpha=alpha

    def cal_cen(self,j):
        m=0
        n=0
        for i in range(self.row):
            m=m+ np.power(self.u[i,j],self.alpha) * data[i,0:4]
            n=n+ np.power(self.u[i,j],self.alpha)
        d=m/n
        return d


    def cal_cost(self):

        Ju=0
        for j in range(c):
            for i in range(self.row):
                Ju=Ju+np.power(self.u[i,j],self.alpha)*np.sum(np.square(data[i,0:4]-self.cal_cen(j)))
        return Ju


    def cal_u(self,i,j,k):
        s=0
        d_ij=data[i:0:4]-self.cal_cen(j)
        d_ik=data[i:0:4]-self.cal_cen(k)
        for k in range(c):
            s=s+np.power(d_ij/d_ik,2/(alpha-1))
        s=1/s




    

if __name__ == '__main__':
    dat=load_iris()
    data = np.array(dat.data[:,0:4])
    print(data)
    c=3
    alpha = 2
    center=np.zeros(c)
    a=fcm(data,c,2)
    activator = True
    U=np.zeros((data.shape[0],c))
    while(activator):
        for j in range(c):
            center[j] = a.cal_cen(j)
        J = a.cal_cost()
        if J < 0.001:
            break
        for i in range(data.shape[0]):
            for j in range(c):
                U[i,j] = a.cal_u(i, j)


    print(data)
