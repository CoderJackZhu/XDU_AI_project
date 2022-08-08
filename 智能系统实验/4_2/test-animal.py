import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torch.nn.functional as F
import matplotlib.pyplot as plt
import cv2
from net_model import ICNET
from net_model import RESNET
from net_model import Net1
from torch.utils.data import TensorDataset
from torch.utils.data import DataLoader
from torchvision import transforms
import os

transform = transforms.Compose([transforms.Resize((224,224)), transforms.ToTensor()])

testset = torchvision.datasets.ImageFolder(root='./data/test', transform=transform)
test_loader = torch.utils.data.DataLoader(testset, batch_size=4, shuffle=False, num_workers=4)

def test(test_loader, model, criterion, device):
    model.eval()
    test_loss = 0
    correct = 0

    with torch.no_grad():
        for data, target in test_loader:
            data, target = data.to(device), target.to(device)
            output = model(data)
            test_loss += criterion(output, target).item()
            pred = output.argmax(dim=1, keepdim=True)
            correct += pred.eq(target.view_as(pred)).sum().item()
        test_loss /= len(test_loader.dataset)
        print('\nTest: Average loss: {:.4f}, Accuracy: {}/{} ({:.2f}%)\n'.format(
            test_loss, correct, len(test_loader.dataset),
            100. * correct / len(test_loader.dataset)))

if __name__ == '__main__':
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = RESNET().to(device)
    criterion = nn.CrossEntropyLoss()
    PATH='./check_point/best_point.pth'
    if os.path.exists(PATH):
        checkpoint = torch.load(PATH)
        model.load_state_dict(checkpoint['model'])
        # model=torch.load(PATH)
    test(test_loader, model, criterion, device)








