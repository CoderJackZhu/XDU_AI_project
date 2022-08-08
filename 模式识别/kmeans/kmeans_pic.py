# -*- coding: utf-8 -*-
"""
Created on Sun Nov  1 18:00:00 2020

@author: tremble
"""

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.datasets import load_iris
import random
from sklearn.manifold import TSNE
    



def eucliDist(A,B):
    return np.sqrt(np.sum(np.square(A - B)))

class kmeans_classify:
    
    def __init__(self, data, k):
        row, col = data.shape

        self.row = row
        self.col = col
        self.k=k
        self.data=data
        #print(self.row,self.col,self.k)
        
    
    def kmeans(self):
        data_mean = random.sample(range(self.row), self.k)
        num1 = data_mean[0]
        num2 = data_mean[1]
        num3 = data_mean[2]
        
        mean1 = np.array([self.data[num1,0:4]])
        mean2 = np.array([self.data[num2,0:4]])
        mean3 = np.array([self.data[num3,0:4]])
        #print(mean1,mean2,mean3)
        

        
        test1 = mean1
        test2 = mean2
        test3 = mean3
        
        data_copy = self.data.copy()
        #d=0

        active = True
        while(active):

            data1 = np.zeros((1, 4))
            data2 = np.zeros((1, 4))
            data3 = np.zeros((1, 4))

            data = data_copy
            
            for i in range(self.row):
                distance1 = eucliDist(data[i,0:4], mean1)
                distance2 = eucliDist(data[i,0:4], mean2)
                distance3 = eucliDist(data[i,0:4], mean3)
                #print(distance1)
                if distance1 < distance2 and distance1 < distance3:
                    data1 = np.vstack((data1, data[i,0:4]))
                    #print(data1)
                if distance2 < distance1 and distance2 < distance3:
                    data2 = np.vstack((data2, data[i,0:4]))
                if distance3 < distance1 and distance3 < distance2:
                    data3 = np.vstack((data3, data[i,0:4]))
            #print(data1,data2,data3)
            
            data1 = np.delete(data1, 0, axis=0)
            data2 = np.delete(data2, 0, axis=0)
            data3 = np.delete(data3, 0, axis=0)
            #print(data1,data2,data3)

            mean1 = np.mean(data1, axis=0)
            mean2 = np.mean(data2, axis=0)
            mean3 = np.mean(data3, axis=0)
            #print(mean1, mean2, mean3)
            
            J1 = 0
            for j in range(data1.shape[0]):
                J1 += np.sum(np.square(data1[j] - mean1))

            J2 = 0
            for j in range(data2.shape[0]):
                J2 += np.sum(np.square(data2[j] - mean2))
                
            J3 = 0
            for j in range(data3.shape[0]):
                J3 += np.sum(np.square(data3[j] - mean3))
            
            
        
            if (test1 == mean1).all() and (test2 == mean2).all and (test3 == mean3).all:
            #d=d+1
            #if d==100:
                active = False
            else:
                test1 = mean1
                test2 = mean2
                test3 = mean3
            #print(d)
        self.data1=data1
        self.data2=data2
        self.data3=data3
        
        return data1, data2, data3
                
    def plot_data(self):
       plt.title('k-means')
       plt.show()
       
       
    def plot_2D(self):
        tsne = TSNE(n_components=2,init='pca',random_state=0)
        result1 = tsne.fit_transform(self.data1)
        result2 = tsne.fit_transform(self.data2)
        result3 = tsne.fit_transform(self.data3)

        
        plt.scatter(result1[:,0],result1[:,1],c='r')
        plt.scatter(result2[:,0],result2[:,1],c='b')       
        plt.scatter(result3[:,0],result3[:,1],c='y') 
        plt.show()
       
       
       
        
      
    def ass_kmeans(self):

        data1 = self.data1  
        data2 = self.data2
        data3 = self.data3
        
        s1=np.zeros(data1.shape[0])
        s2=np.zeros(data2.shape[0])
        s3=np.zeros(data3.shape[0])
        
        
        for i in range(data1.shape[0]):
            ai_1=0
            for j in range(data1.shape[0]):
                dis = eucliDist(data1[i,0:4], data1[j,0:4])
                ai_1 += dis
            ai_1 = ai_1/data1.shape[0]
            
            bi_2=0
            for j in range(data2.shape[0]):
                dis = eucliDist(data1[i,0:4], data2[j,0:4])
                bi_2 += dis
            bi_2 = bi_2/data2.shape[0]
            
            bi_3 = 0
            for j in range(data3.shape[0]):
                dis = eucliDist(data1[i,0:4], data3[j,0:4])
                bi_3 += dis
            bi_3 = bi_3/data3.shape[0]
            
            bi=min(bi_2,bi_3)
            
            s1[i] = (bi-ai_1)/(max(bi,ai_1))
        
        
        for i in range(data2.shape[0]):
            ai_2=0
            for j in range(data2.shape[0]):
                dis = eucliDist(data2[i,0:4], data2[j,0:4])
                ai_2 += dis
            ai_2 = ai_2/data2.shape[0]
            
            bi_1=0
            for j in range(data1.shape[0]):
                dis = eucliDist(data2[i,0:4], data1[j,0:4])
                bi_1 += dis
            bi_1 = bi_1/data1.shape[0]
            
            bi_3 = 0
            for j in range(data3.shape[0]):
                dis = eucliDist(data2[i,0:4], data3[j,0:4])
                bi_3 += dis
            bi_3 = bi_3/data3.shape[0]
            
            bi=min(bi_3,bi_1)
            
            s2[i] = (bi-ai_2)/(max(bi,ai_2))
            
            
        for i in range(data3.shape[0]):
            ai_3=0
            for j in range(data3.shape[0]):
                dis = eucliDist(data3[i,0:4], data3[j,0:4])
                ai_3 += dis
            ai_3 = ai_3/data3.shape[0]
            
            bi_1=0
            for j in range(data1.shape[0]):
                dis = eucliDist(data3[i,0:4], data1[j,0:4])
                bi_1 += dis
            bi_1 = bi_1/data1.shape[0]
            
            bi_2 = 0
            for j in range(data2.shape[0]):
                dis = eucliDist(data3[i,0:4], data2[j,0:4])
                bi_2 += dis
            bi_2 = bi_2/data2.shape[0]
            
            bi=min(bi_2,bi_1)
            
            s3[i] = (bi-ai_3)/(max(bi,ai_3))    
        
            
        
        s1 = np.mean(s1)
        s2 = np.mean(s2)
        s3 = np.mean(s3)
        s=s1+s2+s3
        s=s/3
        return s
            
    
    
if __name__ == '__main__':
    
    dat = np.array(pd.read_csv(r"H:\Datasets\Iris.csv",header = None))
    dat = np.array(dat[:,0:4],dtype=np.float32)
    a=kmeans_classify(dat, 3)
    dat1,dat2,dat3 =a.kmeans()
    #print(dat1,dat2,dat3)
    #a.plot_data()
    print('轮廓系数为')
    s = a.ass_kmeans()
    print(s)
    a.plot_2D()
    
    

    