# -*- coding:utf-8 -*-
# Author : JackZhu

# Data : 2021/5/13 19:28

import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import pandas as pd
import numpy as np
from matplotlib import cm
from mpl_toolkits.mplot3d import Axes3D

DNA_SIZE = 31  # DNA大小
POP_SIZE = 1000  # 种群大小
CROSSOVER_RATE = 0.8  # 交叉概率
MUTATION_RATE = 0.01  # 变异概率
N_GENERATIONS = 60  # 迭代次数
X_BOUND = [-100, 100]
Y_BOUND = [-100, 100]


def F(x, y):
    return x ** 2 + y ** 2


class GA():
    def __init__(self, F):
        self.pop = np.random.randint(2, size=(POP_SIZE, DNA_SIZE * 2))
        self.F = F

    def fit(self):
        x, y = self.decode()
        pred = F(x, y)
        return np.abs(pred - np.max(pred)) + 1e-3

    def select(self,fitness):
        idx = np.random.choice(np.arange(POP_SIZE), size=POP_SIZE, replace=True,
                               p=(fitness) / (fitness.sum()))
        return self.pop[idx]

    def crossover_and_mutation(self, CROSSOVER_RATE=0.8):
        new_pop = []
        for father in self.pop:  # 遍历种群中的每一个个体，将该个体作为父亲
            child = father  # 孩子先得到父亲的全部基因（这里我把一串二进制串的那些0，1称为基因）
            if np.random.rand() < CROSSOVER_RATE:  # 产生子代时不是必然发生交叉，而是以一定的概率发生交叉
                mother = self.pop[np.random.randint(POP_SIZE)]  # 再种群中选择另一个个体，并将该个体作为母亲
                cross_points = np.random.randint(low=0, high=DNA_SIZE * 2)  # 随机产生交叉的点
                child[cross_points:] = mother[cross_points:]  # 孩子得到位于交叉点后的母亲的基因
            self.mutation(child, MUTATION_RATE)  # 每个后代有一定的机率发生变异
            new_pop.append(child)

        return new_pop

    def mutation(self, child, MUTATION_RATE=0.005):
        if np.random.rand() < MUTATION_RATE:  # 以MUTATION_RATE的概率进行变异
            mutate_point = np.random.randint(0, DNA_SIZE * 2)  # 随机产生一个实数，代表要变异基因的位置
            child[mutate_point] = child[mutate_point] ^ 1  # 将变异点的二进制为反转

    def plot(self):
        X = np.linspace(*X_BOUND, 100)
        Y = np.linspace(*Y_BOUND, 100)
        X, Y = np.meshgrid(X, Y)
        Z = self.F(X, Y)
        ax.plot_surface(X, Y, Z, rstride=1, cstride=1,cmap='rainbow')
        ax.set_xlabel('x')
        ax.set_ylabel('y')
        ax.set_zlabel('f(x,y)')
        plt.pause(1)
        plt.show()

    def decode(self):
        x_pop = self.pop[:, 1::2]  # 奇数列表示X
        y_pop = self.pop[:, ::2]  # 偶数列表示y
        # pop:(POP_SIZE,DNA_SIZE)*(DNA_SIZE,1) --> (POP_SIZE,1)
        x = x_pop.dot(2 **np.arange(DNA_SIZE)[::-1]) / float(2 ** DNA_SIZE - 1) * (X_BOUND[1] - X_BOUND[0]) + X_BOUND[0]
        y = y_pop.dot(2 ** np.arange(DNA_SIZE)[::-1]) / float(2 ** DNA_SIZE - 1) * (Y_BOUND[1] - Y_BOUND[0]) + Y_BOUND[0]
        return x, y

    def print_info(self):
        x, y = self.decode()
        pred = self.F(x, y)
        min_prad_index = np.argmin(pred)
        print("最优的基因型：", self.pop[min_prad_index])
        print("x:", x[min_prad_index])
        print("y:", y[min_prad_index])
        print("最小值:", pred[min_prad_index])


if __name__ == '__main__':
    N = 500
    ga = GA(F)
    fig = plt.figure()
    ax = Axes3D(fig)
    plt.ion()
    for i in range(N):
        if (i > N / 2):
            MUTATION_RATE /= 10
        x,y=ga.decode()
        sca = ax.scatter(x, y, F(x, y), c='black', marker='o')
        plt.show()
        plt.pause(0.01)
        sca.remove()
        ga.pop = np.array(ga.crossover_and_mutation(CROSSOVER_RATE))
        fitness=ga.fit()
        ga.pop=ga.select(fitness)

    ga.print_info()
    plt.ioff()
    ga.plot(ax)


