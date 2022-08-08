import cv2
import time
import os
import matplotlib.pyplot as plt
import torch
from torch import nn
import torchvision.models as models
import torchvision.transforms as transforms
import numpy as np
import torch
import torchvision
import torch.optim as optim
import torch.backends.cudnn as cudnn

savepath='./vis_resnet50/features'

# if not os.path.exists(savepath):
#     os.mkdir(savepath)

cudnn.benchmark = True # 对卷积进行加速

def draw_features(width,height,x,savename):
    tic=time.time()
    fig = plt.figure(figsize=(16, 16))
    fig.subplots_adjust(left=0.05, right=0.95, bottom=0.05, top=0.95, wspace=0.05, hspace=0.05)
    for i in range(width*height):
        plt.subplot(height,width, i + 1)
        plt.axis('off')
        # plt.tight_layout()
        img = x[0, i, :, :]
        pmin = np.min(img)
        pmax = np.max(img)
        img = (img - pmin) / (pmax - pmin + 0.000001)
        plt.imshow(img, cmap='gray')
        print("{}/{}".format(i,width*height))
    fig.savefig(savename, dpi=100)
    fig.clf()
    plt.close()
    print("time:{}".format(time.time()-tic))


class ft_net(nn.Module):

    def __init__(self):
        super(ft_net, self).__init__()
        model_ft = models.resnet50(pretrained=True)
        self.model = model_ft

    def forward(self, x):
        if True: # draw features or not
            x = self.model.conv1(x)
            draw_features(8,8,x.cpu().detach().numpy(),"{}/f1_conv1.png".format(savepath))

            x = self.model.bn1(x)
            draw_features(8, 8, x.cpu().detach().numpy(),"{}/f2_bn1.png".format(savepath))

            x = self.model.relu(x)
            draw_features(8, 8, x.cpu().detach().numpy(), "{}/f3_relu.png".format(savepath))

            x = self.model.maxpool(x)
            draw_features(8, 8, x.cpu().detach().numpy(), "{}/f4_maxpool.png".format(savepath))

            x = self.model.layer1(x)
            draw_features(16, 16, x.cpu().detach().numpy(), "{}/f5_layer1.png".format(savepath))

            x = self.model.layer2(x)
            draw_features(16, 32, x.cpu().detach().numpy(), "{}/f6_layer2.png".format(savepath))

            x = self.model.layer3(x)
            draw_features(32, 32, x.cpu().detach().numpy(), "{}/f7_layer3.png".format(savepath))

            x = self.model.layer4(x)
            draw_features(32, 32, x.cpu().detach().numpy()[:, 0:1024, :, :], "{}/f8_layer4_1.png".format(savepath))
            draw_features(32, 32, x.cpu().detach().numpy()[:, 1024:2048, :, :], "{}/f8_layer4_2.png".format(savepath))

            x = self.model.avgpool(x)
            plt.plot(np.linspace(1, 2048, 2048), x.cpu().detach().numpy()[0, :, 0, 0])
            plt.savefig("{}/f9_avgpool.png".format(savepath))
            plt.clf()
            plt.close()

            x = x.view(x.size(0), -1)
            x = self.model.fc(x)
            plt.plot(np.linspace(1, 1000, 1000), x.cpu().detach().numpy()[0, :])
            plt.savefig("{}/f10_fc.png".format(savepath))
            plt.clf()
            plt.close()
        else :
            x = self.model.conv1(x)
            x = self.model.bn1(x)
            x = self.model.relu(x)
            x = self.model.maxpool(x)
            x = self.model.layer1(x)
            x = self.model.layer2(x)
            x = self.model.layer3(x)
            x = self.model.layer4(x)
            x = self.model.avgpool(x)
            x = x.view(x.size(0), -1)
            x = self.model.fc(x)
        return x


def train():
    model.train()
    loss = 0.0
    for batch_idx, (data, target) in enumerate(trainloader):
        data, target = data.to(device), target.to(device)
        optimizer.zero_grad()
        output = model(data)
        loss = criterion(output, target)
        loss.backward()
        optimizer.step()
        if batch_idx % log_interval == 0:
            print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                epoch, batch_idx * len(data), len(trainloader.dataset),
                       100. * batch_idx / len(trainloader), loss.item()))


def test():
    model.eval()
    test_loss = 0
    correct = 0
    with torch.no_grad():
        for data, target in testloader:
            data, target = data.to(device), target.to(device)
            output = model(data)
            test_loss += criterion(output, target).item()
            pred = output.argmax(dim=1, keepdim=True)
            correct += pred.eq(target.view_as(pred)).sum().item()

    test_loss /= len(testloader.dataset)
    print('\nTest set: Average loss: {:.4f}, Accuracy: {}/{} ({:.2f}%)\n'.format(
        test_loss, correct, len(testloader.dataset),
        100. * correct / len(testloader.dataset)))
    return 100. * correct / len(testloader.dataset), model.state_dict()


transform = transforms.Compose(
    [transforms.ToTensor(),
     transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])

trainset = torchvision.datasets.CIFAR10('./data',
                                      download=True,
                                      train=True,
                                      transform=transform)
testset = torchvision.datasets.CIFAR10('./data',
                                     download=True,
                                     train=False,
                                     transform=transform)

trainloader = torch.utils.data.DataLoader(trainset, batch_size=64,
                                          shuffle=True, num_workers=0)

testloader = torch.utils.data.DataLoader(testset, batch_size=64,
                                         shuffle=False, num_workers=0)

log_interval = 300
epochs = 5

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
torch.manual_seed(1)
criterion = nn.CrossEntropyLoss()
model = ft_net().to(device)
model = nn.DataParallel(model)

# optimizer=torch.optim.SGD(model.parameters(),lr=lr,momentum=momentum)
optimizer = optim.Adadelta(model.parameters(), lr=1.0)

if __name__ == '__main__':
    max_acc = float('0')
    for epoch in range(1, epochs + 1):
        train()
        accuracy, net_dict = test()
        if accuracy > max_acc:
            max_acc = accuracy
            torch.save({'epoch_record': epoch, 'model': net_dict},
                       f'./网络特征提取模型_{max_acc}%.pth')