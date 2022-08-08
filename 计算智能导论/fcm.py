import numpy as np
from sklearn import metrics
from sklearn.datasets import load_iris
import random
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE
from sklearn.decomposition import PCA


class FuzzyCMeans():
    def __init__(self, data, c, alpha=2):
        self.alpha = alpha
        self.data = data
        self.c = c
        self.row, self.col = data.shape
        self.matrix = np.zeros((self.row, self.c))
        for i in range(self.row):
            for j in range(self.c-1):
                if np.sum(self.matrix[i, :]) < 1 :
                    self.matrix[i, j] = random.uniform(0, 1 - np.sum(self.matrix[i, :]))
            self.matrix[i, self.c - 1] = 1 - np.sum(self.matrix[i, :])
        self.centers = np.zeros((self.c, self.col))

    def fit(self):
        for j in range(self.c):
            up1 = 0
            down1 = 0
            for i in range(self.row):
                up1 += (self.matrix[i, j] ** self.alpha) * self.data[i]
                down1 += self.matrix[i, j] ** self.alpha
            self.centers[j] = up1 / down1

    def cost(self):
        sum = 0
        for j in range(self.c):
            for i in range(self.row):
                sum += (self.matrix[i, j] ** self.alpha) * (np.linalg.norm(self.data[i] - self.centers[j]) ** 2)
        return sum

    def cal_u(self):
        for i in range(self.row):
            for j in range(self.c):
                down2 = 0
                for k in range(self.c):
                    down2 += (np.linalg.norm(self.data[i] - self.centers[j]) / np.linalg.norm(
                        self.data[i] - self.centers[k])) ** (2 / (self.alpha - 1))

                self.matrix[i, j] = 1 / (down2)

    def cal_label(self):
        lab = np.argmax(self.matrix, axis=1)
        return lab

    def calcute(self, epochs):
        for epoch in range(epochs):
            self.fit()
            result = self.cost()
            print(result)
            self.cal_u()
        label = self.cal_label()
        return label

    def imshow(self, label):
        tsne = TSNE(n_components=2, learning_rate=100).fit_transform(self.data)
        pca = PCA().fit_transform(self.data)
        plt.figure(figsize=(12, 6))
        plt.subplot(121)
        plt.scatter(tsne[:, 0], tsne[:, 1], c=label)
        plt.title('t-SNE')
        plt.subplot(122)
        plt.scatter(pca[:, 0], pca[:, 1], c=label)
        plt.title('PCA')
        plt.colorbar()
        plt.show()


if __name__ == '__main__':
    c = 3
    iris = load_iris()
    data = iris.data
    # target = iris.target
    fcm = FuzzyCMeans(data, c=c)
    label = fcm.calcute(epochs=50)
    print(label)
    fcm.imshow(label)
    s = metrics.silhouette_score(data, label, metric='euclidean')
    print('轮廓系数为{:.4f}'.format(s))
