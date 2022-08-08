import numpy
import pandas as pd
import numpy as np
from scipy.io import loadmat
import matplotlib.pyplot as plt
from sklearn.metrics.pairwise import euclidean_distances


def split(data, MinPts, eps):
    # 划分属性
    ptses = []
    dist = euclidean_distances(data)
    for row in dist:
        # 密度,空间中任意一点的密度是以该点为圆心、以 Eps 为半径的圆区域内包含的点数
        density = numpy.sum(row < eps)
        if density > MinPts:
            # 核心点（Core Points）
            pts = 1
        elif density > 1:
            # 边界点（Border Points）
            pts = 2
        else:
            # 噪声点（Noise Points）
            pts = 0
        ptses.append(pts)
    # 过滤噪声点，因为其无法聚类，自成一类
    corePoints = data[pd.Series(ptses) != 0]
    coreDist = euclidean_distances(corePoints)
    return coreDist


def DBSCAN(dataset, points, radius):
    # Step1. 把每个点的领域都作为一类
    cluster = dict()
    i = 0
    coreDistance = split(dataset, points, radius)
    for row in coreDistance:
        cluster[i] = numpy.where(row < radius)[0]
        i = i + 1
    # Step2. 将有交集的领域，都合并为新的领域
    for i in range(len(cluster)):
        for j in range(len(cluster)):
            if len(set(cluster[j]) & set(cluster[i])) > 0 and i != j:
                cluster[i] = list(set(cluster[i]) | set(cluster[j]))
                cluster[j] = list()
    # Step3. 找出独立（也就是没有交集）的领域，即聚类的结果
    result = dict()
    j = 0
    for i in range(len(cluster)):
        if len(cluster[i]) > 0:
            result[j] = cluster[i]
        j = j + 1
    return result


if __name__ == '__main__':
    # 导入数据
    data_ori = loadmat("./data-密度聚类/long.mat")
    data_info = np.array(data_ori['long1'])
    data = pd.DataFrame(data=data_info[:, :-1], columns=['x', 'y'])
    plt.scatter(data['x'],data['y'])
    # plt.show()

    eps = [0.01, 0.1, 1, 100]
    MinPts = [2, 3, 4, 5]

    # for i in range(len(eps)):
    #     for j in range(len(MinPts)):
    #         ans = DBSCAN(data,MinPts[j],eps[i])
    #         # 找出每个点所在领域的序号，作为他们最后聚类的结果标记
    #         for k in range(len(ans)):
    #             for p in ans[k]:
    #                 data.at[p, 'type'] = k
    #         # 画出聚类结果
    #         plt.scatter(data['x'],data['y'],c=data['type'])
    #         plt.show()
    ans = DBSCAN(data, 5, 0.2)
    # 找出每个点所在领域的序号，作为他们最后聚类的结果标记
    print(ans.keys())
    for k in range(len(ans)):

        for p in ans.keys():
            data.at[p, 'type'] = k
    # 画出聚类结果
    plt.scatter(data['x'], data['y'], c=data['type'])
    plt.show()
