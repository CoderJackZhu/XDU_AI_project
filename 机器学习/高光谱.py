import pandas as pd
import numpy as np
from scipy.io import loadmat
import sklearn
from sklearn import preprocessing
from sklearn.ensemble import RandomForestClassifier as RFC
from sklearn.feature_selection import SelectFromModel
from sklearn.feature_selection import VarianceThreshold
from sklearn.feature_selection import SelectKBest, f_classif
from sklearn.linear_model import LogisticRegression
from sklearn.feature_selection import RFE
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix
from sklearn import metrics


def get_data():
    dat = loadmat('./高光谱数据集/KSC.mat')['KSC']
    lab = loadmat('./高光谱数据集/KSC_gt.mat')['KSC_gt']

    dat = dat.reshape(-1, 176)
    lab = lab.reshape(-1)
    print(dat.shape)

    data, label = [], []
    for i in range(dat.shape[0]):
        if lab[i].all() != 0:
            data.append(dat[i, :])
            label.append(lab[i])

    data = np.array(data)
    label = np.array(label)
    return data, label


def process(data):
    data = preprocessing.StandardScaler().fit_transform(data)
    print('shape={}'.format(data.shape))
    selector = VarianceThreshold()  # 实例化，不填参数默认方差为0
    data = selector.fit_transform(data)
    print(data.shape)
    median_num = np.median(data)
    data = VarianceThreshold(median_num).fit_transform(data)
    print(data.shape)
    return data


# acc = cross_val_score(KNN(), data, label, cv=5).mean()
# print("accuracy:{},time:{}".format(acc,time.time()-start))

def select_k(data, label, k):
    results = SelectKBest(f_classif, k=k).fit(data, label)
    print(results)
    features = pd.DataFrame({
        "score": results.scores_,
        "pvalue": results.pvalues_,
        "select": results.get_support()
    })
    features.sort_values("score", ascending=False)
    print(features)
    index = results.get_support(indices=True)
    print(index)
    return index


def rfe(data, label, n):
    results = RFE(estimator=LogisticRegression(), n_features_to_select=n)
    print(results)
    results.fit(data, label)
    index = results.get_support(indices=True)
    print(index)
    return index


def rfc(data, label):
    RFC_ = RFC(n_estimators=50, random_state=0)
    X_embedded = SelectFromModel(RFC_, threshold=0.005).fit_transform(data, label)
    result = sklearn.model_selection.cross_val_score(RFC_, X_embedded, label, cv=5).mean()
    print(result)


def select_index_data(index, data, label):
    data_after = []
    for i in index:
        data_after.append(data[:, i])
    data_after = np.array(data_after).transpose()
    print(data_after.shape)
    print(label.shape)
    return train_test_split(data_after, label, test_size=0.3, random_state=1)


def measure_feature(train_data, test_data, train_label, test_label, gamma, c):
    clf = sklearn.svm.SVC(kernel='poly', gamma=gamma, C=c)
    clf.fit(train_data, train_label)
    predict = clf.predict(test_data)
    clf.get_params(deep=True)
    acc = sklearn.metrics.accuracy_score(test_label, predict)
    f1 = sklearn.metrics.f1_score(test_label, predict, average='micro')
    recall = metrics.recall_score(test_label, predict, average='micro')
    precision = metrics.precision_score(test_label, predict, average='micro')
    return acc, f1, recall, precision


if __name__ == '__main__':
    data, label = get_data()
    data = process(data)

    # rfc(data,label)
    index = select_k(data, label, k=50)
    # index = rfe(data, label,n=30)
    train_data, test_data, train_label, test_label = select_index_data(index, data, label)
    print(train_data.shape, test_data.shape, train_label.shape, test_label.shape)
    gamma, c = 0.125, 60
    train_acc, train_f1, train_recall, train_precision = measure_feature(train_data, train_data, train_label,
                                                                         train_label, gamma, c)
    test_acc, test_f1, test_recall, test_precision = measure_feature(train_data, test_data, train_label, test_label,
                                                                     gamma, c)

    print(train_acc, test_acc)
    print(train_f1, test_f1)
    print(train_recall, test_recall)
    print(train_precision, test_precision)
    # print('训练集准确率为{:.4f}，测试集准确率为{:.4f}'.format(train_acc, test_acc))
