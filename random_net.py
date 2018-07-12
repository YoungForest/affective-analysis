import torch.nn as nn
import torch
import random
import liris_dataset
from liris_dataset import getDataLoader
from liris_net import evaluate
from liris_net import device

class RandomNet(nn.Module):

    def __init__(self):
        super(RandomNet, self).__init__()

    def forward(self, x):
        return torch.rand(x.shape[0], 2, device=device) * 2 -1

if __name__ == '__main__':
    trainloader, testloader = getDataLoader()

    net = RandomNet()
    net.to(device)
    for i in range(5):
        evaluate(net, testloader)

        
        
