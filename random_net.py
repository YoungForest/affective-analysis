import torch.nn as nn
import torch
import random
import liris_dataset
from liris_net import evaluate
from liris_net import device

class RandomNet(nn.Module):

    def __init__(self):
        super(RandomNet, self).__init__()

    def forward(self, x):
        return torch.rand(x.shape[0], 2, device=device) * 2 -1

if __name__ == '__main__':
    testset = liris_dataset.getLirisDataset('liris-accede-test-dataset.pkl', train=False)
    testloader = torch.utils.data.DataLoader(testset, batch_size=8, shuffle=False)

    net = RandomNet()
    net.to(device)
    for i in range(5):
        evaluate(net, testloader)

        
        
