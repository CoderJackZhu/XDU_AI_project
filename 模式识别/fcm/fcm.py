
import numpy as np
import random
import copy


class FuzzyCMeans(object):

    def __init__(self, cluster_number, m):
        self.cluster = cluster_number
        self.center = None
        self.u = None
        self.m = m

    @staticmethod
    def distance(center, train):
        result = np.linalg.norm(train - center, axis=1)
        return result

    def fit(self, x, y=None):
        u = np.random.randint(1, 10000, (x.shape[0], self.cluster))
        u = u / u.sum(1)[:, np.newaxis]
        for _ in range(5):
            # 创建它的副本，以检查结束条件
            u_old = copy.deepcopy(u)
            # 计算聚类中心
            distance_matrix = np.zeros((self.cluster, x.shape[0]))
            for j in range(0, self.cluster):
                current_cluster_center = np.zeros(x.shape[1])
                for i in range(0, x.shape[1]):
                    dummy_sum_num = ((u[:, j] ** self.m) * x[:, i]).sum(0)
                    dummy_sum_dum = (u[:, j] ** self.m).sum()
                    # 第i列的聚类中心
                    current_cluster_center[i] = dummy_sum_num / dummy_sum_dum
                # 第j簇的所有聚类中心
                distance_matrix[j] = self.distance(x, current_cluster_center)
            distance_matrix = distance_matrix.T
            # 更新U
            for j in range(0, self.cluster):
                for i in range(0, len(x)):
                    dummy = 0.0
                    for k in range(0, self.cluster):
                        # 分母
                        dummy += (distance_matrix[i][j] / distance_matrix[i][k]) ** (2 / (self.m - 1))
                    u[i][j] = 1 / dummy
            if np.linalg.norm(u - u_old) < 1e-7:
                break
        for i in range(0, len(u)):
            maximum = max(u[i])
            for j in range(0, len(u[0])):
                if u[i][j] != maximum:
                    u[i][j] = 0
                else:
                    u[i][j] = 1

        self.u = u
        return self

    def transform(self, x=None):
        indices = self.u.argmax(1)
        out = [x[indices == i] for i in range(indices.max()+1)]
        return out

    def show(self, x=None):
        indices = self.u.argmax(1)
        for i in range(indices.max() + 1):
            x[indices == i] = [50*i, 50*i, 50*i]
        return x


if __name__ == '__main__':
    FuzzyCMeans(3, 2).fit(np.random.randn(200, 3))
