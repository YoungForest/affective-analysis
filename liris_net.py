import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torchvision
from liris_dataset import LirisDataset
from liris_dataset import getDataLoader
import liris_dataset
from torch.utils.data.sampler import SubsetRandomSampler

# Training on GPU
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print(device)

mse_list = []
emotions = ['valence', 'arousal']
# record mse of two emotions every epoch
loss_emotion_epoch = [[], []]

class LirisNet(nn.Module):

    def __init__(self):
        super(LirisNet, self).__init__()
        self.fc1 = nn.Linear(57344, 200)
        self.bn1 = nn.BatchNorm1d(200)
        self.fc2 = nn.Linear(200, 1)
        self.bn2 = nn.BatchNorm1d(1)

    def forward(self, x):
        x = self.bn1(F.relu(self.fc1(x)))
        x = self.bn2(F.relu(self.fc2(x)))

        return x

def evaluate(net, testloader):
    criterion = nn.MSELoss()
    loss_test = 0.0
    for i, data in enumerate(testloader, 0):
        # get the inputs
        inputs = data['input']
        valence = data['labels'][:, 0:1]
        inputs, valence = inputs.to(device), valence.to(device)

        outputs = net(inputs)
        loss = criterion(outputs, valence)
        loss_test += loss.item()

    print('mse average: %f' % (loss_test / len(testloader)))
    mse_list.append(loss_test / len(testloader))

if __name__ == '__main__':
    trainloader, testloader = getDataLoader()

    # new a Neural Network instance
    net = LirisNet()
    if torch.cuda.device_count() > 1:
        print("Let's use", torch.cuda.device_count(), "GPUs!")
        net = nn.DataParallel(net)
    net.to(device)
    # net.load_state_dict(torch.load('nn-2.pth'))
 
    # define a Loss function and optimizer
    criterion = nn.MSELoss()
    optimizer = optim.SGD(net.parameters(), lr=0.0001)
    # print(net.parameters())

    # train the network
    for epoch in range(50): # Loop over the dataset multiple times

        running_loss = 0.0
        for i, data in enumerate(trainloader, 0):
            # get the inputs
            inputs = data['input']
            valence = data['labels'][:, 0:1]
            arousal = data['labels'][:, 1:2]
            inputs, valence, arousal = inputs.to(device), valence.to(device), arousal.to(device)

            # zero the parameter gradients
            optimizer.zero_grad()
            
            # forward + backward + optimize
            outputs = net(inputs)
            loss = criterion(outputs, valence)
            loss.backward()
            optimizer.step()

            # print statistics
            running_loss += loss.item()
            if i % 5 == 4:
                print('[%d. %5d] loss: %.3f' % (epoch + 1, i + 1, running_loss / 5))
                running_loss = 0.0
        # Serialization semantics, save the trained model
        torch.save(net.state_dict(), 'nn-video-only-epoch-%d.pth' %(epoch))


        print('Finished Training')
        evaluate(net, testloader)

    print(mse_list)
