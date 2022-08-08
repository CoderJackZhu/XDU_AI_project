import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt
import cv2
from torch.utils.data import TensorDataset
from torch.utils.data import DataLoader

feature = np.load('test_set.npy')
label = np.load('test_label.npy')


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = torch.load('动物分类.pkl')
feature=torch.tensor(feature).type(torch.FloatTensor)
label=torch.tensor(label).type(torch.FloatTensor)



dataset = TensorDataset(feature, label)
dataloader = DataLoader(dataset, batch_size=4, shuffle=True)

criterion = nn.CrossEntropyLoss()
optimizer = optim.Adadelta(model.parameters(), lr=1.0)

def test():
    correct = 0.0
    with torch.no_grad():
        for data, target in dataloader:
            data, target = data.to(device), target.to(device)
            output = model(data)
            pred = output.argmax(dim=1, keepdim=True)
            correct += pred.eq(target.view_as(pred)).sum().item()
            print(correct)
    print('accuracy:{:.2f}%'.format(100.0 * correct / len(dataloader.dataset)))


if __name__ == '__main__':
    test()