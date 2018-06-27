import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torchvision
from dalc_dataset import DalcDataset

class DalcNet(nn.Module):

    def __init__(self):
        super(DalcNet, self).__init__()
        self.fc1 = nn.Linear(3584, 100)
        self.fc2 = nn.Linear(100, 8)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))

        return x

if __name__ == '__main__':
    net = DalcDataset()

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(net.parameters(), lr=0.001, momentum=0.9)

    dataset = DalcDataset('output-resnet-34-kinetics.json', '/home/data_common/data_yangsen/videos', transform=True, ranking_file='filled-labels_features.csv', sep=',')
    dataloader = torch.util.data.DataLoader(dataset, batch_size=4, shuffle=True, num_workers=2)
    
