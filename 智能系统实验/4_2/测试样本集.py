import numpy as np
import torch
import torch.nn as nn
import matplotlib.pyplot as plt
import cv2
from torch.utils.data import TensorDataset
from torch.utils.data import DataLoader

train_set=np.load('train_set.npy')
test_set = np.load('test_set.npy')
img1=train_set[69,0,:,:]
img2=test_set[30,0,:,:]
img1=img1.astype(np.uint8)
img2=img2.astype(np.uint8)

print(img1)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
label = np.load('test_label.npy')


x=np.load(file='test_set.npy')/255

x=torch.tensor(x).type(torch.FloatTensor).to(device)
y1=torch.zeros(30)
y2=torch.zeros(30)
y0=torch.cat((y1,y2)).type(torch.LongTensor).to(device)

print(test_set.shape)







