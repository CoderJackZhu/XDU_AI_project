import torch
import numpy as np
import torch.nn as nn


class ICNET(nn.Module):
    def __init__(self):
        super(ICNET, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(3,8,kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.Conv2d(8,8,kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2, padding=0)
        )
        self.fc = nn.Sequential(
            nn.Linear(64*64*8,1000),
            nn.ReLU(),
            nn.Linear(1000, 1000),
            nn.ReLU(),
            nn.Linear(1000, 2),
            nn.Softmax(dim=1)
        )

    def forward(self, x):
        x = self.conv(x)
        x = x.view(x.size()[0], -1)
        x = self.fc(x)
        return x


if __name__ == '__main__':
    x = torch.zeros(2,3,224,224).cuda()
    net = ICNET().cuda()
    out = net(x)
    print(out.detach().cpu().numpy())
