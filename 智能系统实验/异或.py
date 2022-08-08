import os
import torch
import numpy as np
import torch.nn as nn


class FCN(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super(FCN, self).__init__()

        self.model = nn.Sequential(
            nn.Linear(input_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, output_size),
            nn.Sigmoid(),
        )

    def forward(self, x):
        x = self.model(x)
        return x


def train():
    # Device configuration
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    # Model
    print("Building model...")
    initial_epoch = 0
    model = FCN(input_size, hidden_size, output_size)
    model.to(device)

    print("Loading data...")
    x = [[0,0],[0,1],[1,0],[1,1]]
    x_tensor = torch.tensor(x).float().cuda()
    y = [[0],[1],[1],[0]]
    y_tensor = torch.tensor(y).float().cuda()

    loss_function = nn.MSELoss(reduce=True, size_average=True)
    optimizer = torch.optim.SGD(model.parameters(), lr=0.05)

    print("Beginning to train...")
    min_loss = float('inf')
    for epoch in range(5000):
        out = model(x_tensor)
        loss = loss_function(out,y_tensor)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        if epoch % 1000 == 0:
            print('Epoch num:', epoch)
            print('Loss:', loss)
        if loss < min_loss:
            min_loss = loss
            torch.save({'epoch_record': epoch, 'model': model.state_dict()},
                       './base_model_path')


def test():
    # Device configuration
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    # Model
    print("Building model...")
    initial_epoch = 0
    model = FCN(input_size, hidden_size, output_size)
    if os.path.exists('./base_model_path'):
        load_dict = torch.load('./base_model_path')
        initial_epoch = load_dict['epoch_record']
        model.load_state_dict(load_dict['model'])
    model.to(device)
    # input
    x = [[1, 0]]
    x_tensor = torch.tensor(x).float().cuda()
    # output
    print("Begin to test...")
    outFinal = model(x_tensor).detach().cpu().numpy()
    if outFinal > 0.5:
        res = 1
    else:
        res = 0
    print('Input:', x)
    print('Answer:', res)


if __name__ == '__main__':
    input_size = 2
    hidden_size = 20
    output_size = 1
    #train()
    test()

