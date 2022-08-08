# -*- coding: utf-8 -*-
"""
Created on Mon Nov  9 21:38:07 2020

@author: tremble
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
plt.rcParams["axes.unicode_minus"] = False
Iris_data = np.array(pd.read_csv(r"H:\iris.csv").values[:,0:4], dtype=np.float32)

def k_mean(iteration, Iris_data, *init):
    iter_ = 0
    n_data, width = Iris_data.shape
    mean = np.mean(Iris_data, axis=0)
    std = np.std(Iris_data, axis=0)
    copy = (Iris_data - mean) / std
    init_value = np.array(init , dtype=np.float32)
    init_value = (init_value - mean) / std
    k, _ = init_value.shape
    accuracy_list = []
    while iter_ < iteration:
        label = np.array([5 for _ in range(n_data)])
        ori_label = []
        for i in range(3):
            ori_label.extend([i for _ in range(50)])
        sample_belong_array = np.zeros((k, k), dtype=np.int32)
        for i in range(n_data):
            result = []
            for j in range(k):
                result.append((np.linalg.norm(copy[i, :]-init_value[j, :]), j))
            belong = min(result, key=lambda x: x[0])[1]
            label[i] = belong
        result = np.zeros((k,4))
        for i in range(k):
            return_ = np.where(label == i)
            if len(return_) == 0:
                pass
            else:
                result[i] = np.mean(copy[return_],axis=0)
        init_value = result
        iter_ += 1
        for i in range(len(label)):
            sample_belong_array[ori_label[i], label[i]] += 1
        accuracy1 = np.trace(sample_belong_array) / n_data
        P = sample_belong_array / np.sum(sample_belong_array, axis=0)
        E = - np.sum(np.sum(P * np.log2(P+1e-8), axis=0) * np.sum(sample_belong_array, axis=0)/150)
        accuracy_list.append((accuracy1, E))
    plt.figure()
    plt.plot([i+1 for i in range(iteration)], np.array(accuracy_list)[:, 0], label="accuracy")
    plt.plot([i+1 for i in range(iteration)], np.array(accuracy_list)[:, 1], label="entropy")
    plt.legend()

def visual():
    m = np.mean(Iris_data, axis=0)
    std = np.std(Iris_data, axis=0)
    Z = (Iris_data - m) / std
    value, vector = np.linalg.eig(np.dot(Z.T, Z))
    index = np.argpartition(value, -2)[-2:]
    w = vector[:, index]
    Y = np.dot(Iris_data, w)
    plt.figure()
    plt.scatter(Y[:50, 0], Y[:50, 1], label="class_1",c='r')
    plt.scatter(Y[50:100, 0], Y[50:100, 1], label="class_2",c='y')
    plt.scatter(Y[100:150, 0], Y[100:150, 1], label="class_3",c='b')
    plt.legend()

if __name__ == "__main__":
    k_mean(100, Iris_data, list(Iris_data[25]), list(Iris_data[75]), list(Iris_data[125]))
    visual()
    plt.show() 
