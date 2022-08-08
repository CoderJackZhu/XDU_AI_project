import sklearn
import torch
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
from PIL import Image
import numpy as np
from skimage import feature as ft
import pandas as pd
import sklearn
import random
from sklearn import datasets
from sklearn.preprocessing import scale
from sklearn.manifold import TSNE
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
from sklearn.metrics import silhouette_score
from sklearn import svm
from sklearn.cluster import KMeans
from sklearn import metrics

transform = transforms.Compose(
    [transforms.ToTensor(),
     transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])

trainset = torchvision.datasets.CIFAR10(root='./data', train=True,
                                        download=True, transform=transform)
# trainloader = DataLoader(trainset, batch_size=64,
#                          shuffle=True, num_workers=2)

testset = torchvision.datasets.CIFAR10(root='./data', train=False,
                                       download=True, transform=transform)
# testloader = DataLoader(testset, batch_size=64,
#                         shuffle=False, num_workers=2)

train = trainset.data.transpose(0, 3, 1, 2)
test = testset.data.transpose(0, 3, 1, 2)


def hog_extraction(data, size=1):
    """
    功能：提取图像HOG特征

    输入：
    data:(numpy array)输入数据[num,3,32,32]
    size:(int)(size,size)为提取HOG特征的cellsize

    输出：
    data_hogfeature:(numpy array):data的HOG特征[num,dim]
    """
    num = data.shape[0]
    data = data.astype('uint8')
    # 提取训练样本的HOG特征
    data1_hogfeature = []
    for i in range(num):
        x = data[i]
        r = Image.fromarray(x[0])
        g = Image.fromarray(x[1])
        b = Image.fromarray(x[2])

        # 合并三通道
        img = Image.merge("RGB", (r, g, b))
        # 转为灰度图
        gray = img.convert('L')
        #        out=gray.resize((100,100),Image.ANTIALIAS)
        # 转化为array
        gray_array = np.array(gray)
        # 提取HOG特征
        hogfeature = ft.hog(gray_array, pixels_per_cell=(size, size))
        print(hogfeature)
        data1_hogfeature.append(hogfeature)

    # 把data1_hogfeature中的特征按行堆叠
    data_hogfeature = np.reshape(np.concatenate(data1_hogfeature), [num, -1])
    return data_hogfeature


class Kmeans():
    def __init__(self, dat, k):
        data = scale(dat)
        self.data = data
        self.row, self.col = data.shape
        self.k = k
        self.centers = np.ndarray((k, self.col))
        choices = random.choices(range(self.row), k=k)
        for i in range(k):
            self.centers[i, :] = self.data[choices[i], :]

    def fit(self):
        count = 0
        while (count < 15):
            self.labels = np.zeros((self.row))
            for i in range(self.data.shape[0]):
                dis = []
                for j in range(self.k):
                    dis.append(np.linalg.norm(self.data[i, :] - self.centers[j, :], axis=0))
                lab = np.argmin(dis, axis=0)
                self.labels[i] = lab
            self.result = {}
            for i in range(self.k):
                type = np.where(self.labels == i)[0]
                self.result[i] = type
                if len(type) == 0:
                    self.centers[i, :] = 0
                else:
                    self.centers[i, :] = np.mean(self.data[type, :], axis=0)
            count += 1
        return self.centers, self.result, self.labels

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


hog_train = hog_extraction(train, size=4)
hog_val = hog_extraction(test, size=4)

# kmeans=Kmeans(hog_val,10)
# centers,results,labels=kmeans.fit()
# kmeans.imshow()
# s = silhouette_score(hog_val, testset.targets)
# print(centers)
# print(results)
# print(s)



clf=svm.SVC()
#训练
clf.fit(hog_val,testset.targets)
#预测验证集类别
result=clf.predict(hog_val)
#计算验证集精度
score=clf.score(hog_val,testset.targets)
print(result)
print(score)
