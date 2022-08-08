import torch
import cv2 as cv
import numpy as np

'''
构建样本集
'''
m1 = 70
m2 = 30
l_x = 128
train_set = np.zeros(m1*2*l_x*l_x*3)
train_set = np.reshape(train_set,(m1*2,3,l_x,l_x))
test_set = np.zeros(m2*2*l_x*l_x*3)
test_set = np.reshape(test_set,(m2*2,3,l_x,l_x))

# 猫为0，狗为1
success_mark = 0
train_label = torch.cat((torch.zeros(70),torch.ones(70)))
for i in range(m1):
    path1 = f'./sample/cat.{i}.jpg'
    path2 = f'./sample/dog.{i}.jpg'
    img1 = cv.resize(cv.imread(path1), (l_x,l_x))
    img2 = cv.resize(cv.imread(path2), (l_x,l_x))
    train_set[i,0,:,:] = img1[:,:,0]
    train_set[i,1,:,:] = img1[:,:,1]
    train_set[i,2,:,:] = img1[:,:,2]
    success_mark+=1
    train_set[m1+i,0,:,:] = img2[:,:,0]
    train_set[m1+i,1,:,:] = img2[:,:,1]
    train_set[m1+i,2,:,:] = img2[:,:,2]
    success_mark+=1

for i in range(m2):
    path1 = f'./sample/cat.{i+m1}.jpg'
    path2 = f'./sample/dog.{i+m1}.jpg'
    img1 = cv.resize(cv.imread(path1), (l_x,l_x))
    img2 = cv.resize(cv.imread(path2), (l_x,l_x))
    test_set[i,0,:,:] = img1[:,:,0]
    test_set[i,1,:,:] = img1[:,:,1]
    test_set[i,2,:,:] = img1[:,:,2]
    success_mark+=1
    test_set[m2+i,0,:,:] = img2[:,:,0]
    test_set[m2+i,1,:,:] = img2[:,:,1]
    test_set[m2+i,2,:,:] = img2[:,:,2]
    success_mark+=1

print('Split the data: Done!')
if success_mark == 200:
    np.save('./catdog_train_set224.npy',train_set)
    np.save('./catdog_test_set224.npy',test_set)