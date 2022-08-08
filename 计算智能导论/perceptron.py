import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import torch

class Perceptron(object):
    def __init__(self, input_num, activator_fun):
        self.activator = activator_fun
        self.weights = [0.0] * input_num
        self.bias = 0.0
        print("initial weight:{0}, bias:{1}".format(self.weights, self.bias))

    def __str__(self):
        return 'weights: {0}   ' \
               'bias: {1}\n'.format(self.weights, self.bias)

    def predict(self, input_vec):
        zipped = list(zip(input_vec, self.weights))
        sum_total = sum(list(map(lambda x_y: x_y[0] * x_y[1], zipped)))
        return self.activator(sum_total + self.bias)

    def train(self, input_vecs, labels, iteration, rate):
        for i in range(iteration):
            self._one_iteration(input_vecs, labels, rate)

    def _one_iteration(self, input_vecs, labels, rate):
        samples = list(zip(input_vecs, labels))
        for (input_vec, label) in samples:
            output = self.predict(input_vec)
            self._update_weights(input_vec, output, label, rate)

    def _update_weights(self, input_vec, output, label, rate):
        delta = label - output
        self.weights = list(map(
            lambda x_w: rate * delta * x_w[0] + x_w[1],
            zip(input_vec, self.weights)))
        self.bias += rate * delta

def f_active_function(x):
    return 1 if x > 0 else 0


def get_training_dataset():
    data = torch.ones(100, 2)
    x0 = torch.normal(2 * data, 1)
    x1 = torch.normal(-2 * data, 1)
    x = torch.cat((x0, x1), 0).type(torch.FloatTensor)
    y0 = torch.zeros(100)
    y1 = torch.ones(100)
    y = torch.cat((y0, y1)).type(torch.LongTensor)
    x_ls, y_ls = [], []
    for i in range(100):
        x_ls.append(x[i])
        y_ls.append(y[i])
    return x_ls,y_ls




if __name__ == '__main__':

    data, target = get_training_dataset()
    p = Perceptron(len(data), f_active_function)
    p.train(data, target, 1000, 0.1)
    print(p)
    acc=0
    for i in range(len(data)):
        result=p.predict(data[i])
        print(result)
        acc+=(result==target[i])
    print(acc/len(data))


