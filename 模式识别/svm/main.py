from sklearn import svm
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import normalize
import os
import cv2
import numpy as np
from sklearn import preprocessing

def process(path):
    data = []
    label = []

    packages = os.listdir(path)
    for pack in packages:
        label = int(pack[1:])

        dir = os.path.join(path, pak)
        faces = os.listdir(dir)
        for face in faces:
            face_dir = os.path.join(dir, face)
            pic = cv2.imread(face_dir, 0)
            arr = np.array(pic)
            arr = cv2.resize(arr, (50, 50))
            data.append(arr)
            label.append(label)

    data = np.array(data).reshape(len(labels), -1)
    data = preprocessing.scale(data)

    labels = np.array(labels)

    return data, labels

def main():

    path='./YALE'

    datas, labels = process(path)


    data_train, data_test, label_train, label_test = train_test_split(data, label,test_size=0.3,random_state=1,shuffle=True)
    print(data_train)


    rbf_svm = svm.SVC(C=1.5,kernel='rbf',decision_function_shape='ovr')
    linear_svm = svm.SVC(C=1,kernel = 'poly',decision_function_shape = 'ovr',degree= 3)


    rbf_svm.fit(data_train,label_train)
    linear_svm.fit(data_train,label_train)

    rbf_predict = rbf_svm.predict(data_test)
    linear_svm = linear_svm.predict(data_test)


    acc1 = sum(rbf_predict == label_test)/len(label_test)
    acc2 = sum(linear_svm == label_test )/len(label_test)

    print('accuracy1',acc1)
    print('accuracy2',acc2)

if __name__ =='__main__':
    main()