# -*- coding:utf-8 -*-
# Author : JackZhu

# Data : 2021/5/9 15:16
# 调用科学计算包与绘图包
import numpy as np
import random
import matplotlib.pyplot as plt
import scipy.io as scio
import matplotlib.colors as mcolors
from sklearn import metrics
import sklearn
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import silhouette_score, calinski_harabasz_score, davies_bouldin_score


def calDist(X1, X2):
    sum = 0
    for x1, x2 in zip(X1, X2):
        sum += (x1 - x2) ** 2
    return sum ** 0.5


def getNeibor(data, dataSet, e):
    res = []
    for i in range(dataSet.shape[0]):
        if calDist(data, dataSet[i]) < e:
            res.append(i)
    return res


def DBSCAN(dataSet, e, minPts):
    coreObjs = {}  # 初始化核心对象集合
    C = {}
    n = dataSet.shape[0]
    # 找出所有核心对象，key是核心对象的index，value是ε-邻域中对象的index
    for i in range(n):
        neibor = getNeibor(dataSet[i], dataSet, e)
        if len(neibor) >= minPts:
            coreObjs[i] = neibor
    oldCoreObjs = coreObjs.copy()
    k = 0  # 初始化聚类簇数
    notAccess = list(range(n))  # 初始化未访问样本集合（索引）
    while len(coreObjs) > 0:
        OldNotAccess = []
        OldNotAccess.extend(notAccess)
        cores = coreObjs.keys()
        # 随机选取一个核心对象
        randNum = random.randint(0, len(cores) - 1)
        cores = list(cores)
        core = cores[randNum]
        queue = []
        queue.append(core)
        notAccess.remove(core)
        while len(queue) > 0:
            q = queue[0]
            del queue[0]
            if q in oldCoreObjs.keys():
                delte = [val for val in oldCoreObjs[q] if val in notAccess]  # Δ = N(q)∩Γ
                queue.extend(delte)  # 将Δ中的样本加入队列Q
                notAccess = [val for val in notAccess if val not in delte]  # Γ = Γ\Δ
        k += 1
        C[k] = [val for val in OldNotAccess if val not in notAccess]
        for x in C[k]:
            if x in coreObjs.keys():
                del coreObjs[x]
    return C


def draw(C, D):
    colors = list(mcolors.TABLEAU_COLORS.keys())
    predict = np.zeros((D.shape[0], D.shape[1] + 1))
    j = 0
    keys = C.keys()
    print(keys)
    for k in keys:
        for i in C[k]:
            predict[j, 0:2] = D[i]
            predict[j, 2] = k
            j = j + 1
            plt.scatter(D[i, 0], D[i, 1], color=colors[k + 1])
    plt.show()
    return predict


def main():
    path = './data-密度聚类/square1.mat'
    data = scio.loadmat(path)['square1']
    # plt.scatter(data[:,0],data[:,1])
    # plt.show()
    D = data[:, 0:2]
    label = data[2]
    C = DBSCAN(D, 0.9, 15)
    predict = draw(C, D)
    s1 = metrics.silhouette_score(predict[:, 0:2], predict[:, 2], metric='euclidean')
    s2 = calinski_harabasz_score(predict[:, 0:2], predict[:, 2])  # 计算CH score
    s3 = davies_bouldin_score(predict[:, 0:2], predict[:, 2])  # 计算 DBI

    print(s1, s2, s3)


if __name__ == '__main__':
    main()
