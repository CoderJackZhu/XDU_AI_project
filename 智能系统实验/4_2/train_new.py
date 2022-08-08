import os
import cv2
import torch
import torchvision
import numpy as np
from time import time
import torch.nn as nn
import torch.nn.functional as F
import matplotlib.pyplot as plt
import torchvision.transforms as transforms
from torch.utils.tensorboard import SummaryWriter

def data_loader():
    """加载数据并数据增强"""
    train_transforms = transforms.Compose([
        transforms.Resize((224,224)),
        transforms.RandomHorizontalFlip(), # 随机水平翻转
        transforms.RandomGrayscale(),      # 随机灰度
        # transforms.Grayscale(1),
        transforms.ToTensor()              # 数据类型转换为tensor,并归一化到[0,1]
    ])
    test_transforms = transforms.Compose([transforms.Resize((224,224)),
                                          # transforms.Grayscale(1),
                                          transforms.ToTensor()])
    # cat_dog_dataset = CDDataset()
    trainset = torchvision.datasets.ImageFolder(root='./data/train',transform=train_transforms)
    trainloader = torch.utils.data.DataLoader(trainset, batch_size=32,shuffle=True, num_workers=4)
    testset = torchvision.datasets.ImageFolder(root='./data/test',transform=test_transforms)
    testloader = torch.utils.data.DataLoader(testset, batch_size=4,shuffle=True, num_workers=4)

    # 显示数据信息
    print(len(trainset),len(testset))
    return trainloader,testloader

def imshow(dataloader):
    """可视化数据"""
    classes_names = ['cat', 'dog']

    plt.figure(figsize=(4,4))
    dataiter = iter(dataloader)
    features,labels = dataiter.next()
    print(features.shape,labels)
    for i in range(len(labels)):
        plt.subplot(4,4,i+1)
        plt.xticks([])
        plt.yticks([])
        plt.grid(False)
        npimg = features[i].numpy()
        plt.imshow(np.transpose(npimg, (1, 2, 0)))
        plt.xlabel(classes_names[labels[i]])
    plt.show()

class Net(nn.Module):
    def __init__(self):
        super().__init__()
        ##########################model 1##################################
        # self.conv1 = nn.Conv2d(1, 32, 20, 4)   # n*1*128*128-->n*32*28*28
        # self.norm1 = nn.BatchNorm2d(32)
        # nn.init.xavier_uniform_(self.conv1.weight)
        # # MaxPool的移动步长默认为kernel_size
        # self.maxpool1 = nn.MaxPool2d(kernel_size=2)  # n*32*28*28-->n*32*14*14
        #
        # self.conv2 = nn.Conv2d(32, 64, 3, 1, 2)     # n*32*14*14-->n*64*16*16
        # self.norm2 = nn.BatchNorm2d(64)
        # nn.init.xavier_uniform_(self.conv2.weight)
        # self.maxpool2 = nn.MaxPool2d(2, 2)          # n*64*16*16-->n*64*8*8
        #
        #
        # self.fc1 = nn.Linear(4096,4096)             # n*4096-->n*4096
        # self.fc2 = nn.Linear(4096, 10)              # n*4096-->n*10

        ##########################model 2#####################################
        self.model = torchvision.models.resnet34(pretrained=True)
        self.model.fc = nn.Linear(512,2)

    def forward(self, x):
        # out = F.relu(self.conv1(x))
        # out = self.norm1(out)
        # out = self.maxpool1(out)
        #
        # out = F.relu(self.conv2(out))
        # out = self.norm2(out)
        # out = self.maxpool2(out)
        #
        # out = out.view(out.size(0),-1)
        # out = F.relu(self.fc1(out))
        # out = self.fc2(out)
        # return out
        return self.model(x)

    def evaluate_acc(self,criterion,test_iter, device):
        """模型测试"""
        acc_sum, n, test_loss,batch_cont = 0.0, 0,0.0,0
        with torch.no_grad():
            for X, y in test_iter:
                self.eval()
                acc_sum += (self(X.to(device)).argmax(dim=1) == y.to(device)).sum().item()
                loss = criterion(self(X.to(device)), y.to(device))
                test_loss += loss.item()
                self.train()
                n += y.shape[0]
                batch_cont += 1
            test_loss = test_loss/batch_cont
            acc_sum = acc_sum/n
        return test_loss,acc_sum

    def train_model(self,device,num_epochs,PATH):
        """模型训练"""
        train_iter,test_iter = data_loader()
        imshow(test_iter)
        writer = SummaryWriter()
        start_epoch = 0
        best_score = 0
        optimizer = torch.optim.Adam(net.parameters(),lr=0.001)
        # 自适应学习率
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode="max",
                                                               factor=0.85, patience=0)
        if os.path.exists(PATH) is not True:
            criterion = nn.CrossEntropyLoss()
        else:
            checkpoint = torch.load(PATH)
            self.load_state_dict(checkpoint['model_state_dict'])
            optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
            start_epoch = checkpoint['epoch']
            criterion = checkpoint['criterion']

        patience_counter = 0
        for epoch in range(start_epoch,num_epochs):

            train_loss, train_acc, n, batch_count,start = 0.0, 0.0, 0, 0,time()

            for features, labels in train_iter:
                features,labels = features.to(device), labels.to(device)
                outputs = self(features)
                loss = criterion(outputs, labels)
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                train_loss += loss.item()
                train_acc += (outputs.argmax(dim=1) == labels).sum().item()
                n += labels.shape[0]
                batch_count += 1

            test_loss, test_acc = self.evaluate_acc(criterion,test_iter,device)
            print('epoch %d, train_loss: %.4f, train acc: %.2f%%, test_loss: %.4f,test acc: %.2f%%, time %.1f sec'
                  % (epoch + 1, train_loss / batch_count, 100 * train_acc / n, test_loss, 100 * test_acc, time() - start))
            writer.add_scalar('train/train_acc', train_acc/n, epoch)
            writer.add_scalar('test/test_acc',test_acc,epoch)
            writer.add_scalar('test/test_loss', test_loss, epoch)
            writer.add_scalar('train/train_loss', train_loss/batch_count, epoch)

            scheduler.step(test_acc)
            if test_acc < best_score:
                patience_counter += 1
            else:
                best_score = test_acc
                patience_counter = 0
                torch.save({"epoch": epoch,
                            "model": net.state_dict(),
                            "best_score": best_score},
                            './check_point/best_point.pth')

            if epoch % 5 == 0:
                torch.save({
                    'epoch': epoch,
                    'model_state_dict': net.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'criterion':criterion
                }, PATH)

            if patience_counter >= 5:
                print("-> Early stopping: patience limit reached, stopping...")
                break
        writer.close()
        print('Finished Training')


if __name__ == '__main__':

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    # device = 'cpu'
    print(device)
    net = Net()
    net = net.to(device)
    num_epochs = 80
    learning_rate = 0.001
    model_path = './check_point/check_point.pth'

    net.train_model(device,num_epochs,model_path)


