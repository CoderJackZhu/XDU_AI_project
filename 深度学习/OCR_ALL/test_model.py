import os
import main
import torch
from main import test
import torch.nn as nn
from dataset import MyDataset
from torchvision import  transforms
from main import Net
import torch.optim as optim
transform = transforms.Compose([
        transforms.Resize((64, 64)),
        transforms.ToTensor(),
        lambda _: _.mean(0).unsqueeze(0),
        transforms.Normalize((0.1307,), (0.3081,)),
    ])
use_cuda = torch.cuda.is_available()
device = torch.device("cuda" if use_cuda else "cpu")

test_dataset = MyDataset("./dataset", "test", transformer = transform)
testloader = torch.utils.data.DataLoader(test_dataset, batch_size=4,
                                         shuffle=False, num_workers=0)

if use_cuda:
    model = nn.DataParallel(Net()).cuda()
else:
    model = Net().to(device)

optimizer = optim.Adadelta(model.parameters(), lr=1.0)
transform = transforms.Compose([
        transforms.Resize((64, 64)),
        transforms.ToTensor(),
        lambda _: _.mean(0).unsqueeze(0),
        transforms.Normalize((0.1307,), (0.3081,)),
    ])
path = '.\good_models'
files = os.listdir(path)
for file in files:
    file_path = os.path.join(path,file)
    print(file_path)
    model_dict=torch.load(file_path)
    model = model.load_state_dict(model_dict)
    test_acc = test(model,device, testloader)
