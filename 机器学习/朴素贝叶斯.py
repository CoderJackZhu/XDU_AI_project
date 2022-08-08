# -*- coding: utf-8 -*-
"""
Created on Sat Apr 10 19:20:42 2021

@author: tremble
"""

import numpy as np
from sklearn.metrics import accuracy_score

class NaiveBayes:

    def __init__(self):
        pass

    def gaussion_pdf(self, x_test, x):
        """定义高斯分布"""
        temp1 = (x_test - x.mean(0)) * (x_test - x.mean(0))
        temp2 = x.std(0) * x.std(0)
        return np.exp(-temp1 / (2 * temp2)) / np.sqrt(2 * np.pi * temp2)

    def fit(self, x_train, y_train):
        """读取训练数据"""
        self.x_train = x_train
        self.y_train = y_train
        return self

    def predict(self, x_test):
        """根据新样本的已有特征在数据集中的条件概率来判断新样本所属类别"""
        assert len(x_test.shape) == 2
        self.classes = np.unique(self.y_train)
        pred_probs = []
        # 对于每个输入,计算其处于每个类别的概率
        for i in self.classes:
            idx_i = self.y_train == i
            # 计算P(y)
            p_y = len(idx_i) / len(self.y_train)
            # 利用高斯概率密度函数计算P(x|y)
            p_x_y = np.prod(self.gaussion_pdf(x_test, self.x_train[idx_i]), 1)
            # 计算x,y的联合概率,P(x|y)P(y)
            prob_i = p_y * p_x_y
            pred_probs.append(prob_i)
        pred_probs = np.vstack(pred_probs)
        # 取具有最高概率的类别
        label_idx = pred_probs.argmax(0)
        y_pred = self.classes[label_idx]
        return y_pred

    def score(self, X_test, y_test):
        """根据测试数据集确定当前模型的准确度"""
        y_predict = self.predict(X_test)
        return accuracy_score(y_test, y_predict)
