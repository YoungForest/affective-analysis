from torchvision import models
import torch.nn as nn
import torch
import torch.nn.functional as F
from liris_dataset import LirisDataset
from liris_dataset import getDataLoader
import liris_dataset
import torch.optim as optim

# Training on GPU
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
# device = torch.device("cpu")
print(device)

mse_list = []
emotions = ['valence', 'arousal']
# record mse of two emotions every epoch
loss_emotion_epoch = [[], []]

def evaluate(net, testloader):
    criterion = nn.MSELoss()
    loss_test = 0.0
    loss_emotion = [0.0] * 2
    for i, data in enumerate(testloader, 0):
        # get the inputs
        inputs = data['mel']
        labels = data['labels']
        inputs.unsqueeze_(1)
        inputs, labels = inputs.to(device), labels.to(device)

        outputs = net(inputs)[0]
        loss = criterion(outputs, labels)
        loss_test += loss.item()

        for i in range(2):
            loss = criterion(outputs[:, i], labels[:, i])
            loss_emotion[i] += loss.item()

    print('test result: ')
    for i in range(2):
        print('%s mse: %.3f' % (emotions[i], loss_emotion[i] / len(testloader)))
        loss_emotion_epoch[i].append(loss_emotion[i] / len(testloader))

    print('mse average: %.3f' % (loss_test / len(testloader)))
    mse_list.append(loss_test / len(testloader))



class AudioNet(nn.Module):

    def __init__(self):
        super(AudioNet, self).__init__()
        # 1 input image channel, 6 output channels, 5x5 square convolution
        # kernel
        self.bn0 = nn.BatchNorm2d(1)
        self.conv1 = nn.Conv2d(1, 6, (1, 5))
        self.bn1 = nn.BatchNorm2d(6)
        self.conv2 = nn.Conv2d(6, 16, (1, 4))
        self.bn2 = nn.BatchNorm2d(16)
        self.conv3 = nn.Conv2d(16, 32, (1, 1))
        self.bn3 = nn.BatchNorm2d(32)
        # an affine operation: y = Wx + b
        self.fc1 = nn.Linear(32 * 10 * 97, 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, 2)

    def forward(self, x):
        # Max poolint over a (2, 2) window
        x = self.bn0(x)
        x = F.max_pool2d(F.relu(self.bn1(self.conv1(x))), 2)
        x = F.max_pool2d(F.relu(self.bn2(self.conv2(x))), (1, 2))
        x = F.max_pool2d(F.relu(self.bn3(self.conv3(x))), (1, 2))
        # x = F.max_pool2d(F.relu(self.conv1(x)), 2)
        # x = F.max_pool2d(F.relu(self.conv2(x)), 2)
        # x = F.max_pool2d(F.relu(self.conv3(x)), 2)
        features = x.view(-1, self.num_flat_features(x))
        x = F.relu(self.fc1(features))
        x = F.relu(self.fc2(x))
        x = F.relu(self.fc3(x))
        return x, features
   
    def num_flat_features(self, x):
        size = x.size()[1:]
        num_features = 1
        for s in size:
            num_features *= s
        return num_features

def main():
    trainloader, testloader = getDataLoader()

    net = AudioNet().cuda()
    net = nn.DataParallel(net)
    # net.load_state_dict(torch.load('nn-epoch-49.pth'))
 
    # define a Loss function and optimizer
    criterion = nn.MSELoss()
    optimizer = optim.SGD(net.parameters(), lr=0.0001)


    # train the network
    for epoch in range(100): # Loop over the dataset multiple times

        running_loss = 0.0
        for i, data in enumerate(trainloader, 0):
            # get the inputs
            inputs = data['mel']
            labels = data['labels']
            inputs.unsqueeze_(1)
            inputs, labels = inputs.to(device), labels.to(device)

            # zero the parameter gradients
            optimizer.zero_grad()

            # forward + backward + optimize
            outputs = net(inputs)[0]
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            # print statistics
            running_loss += loss.item()
            if i % 5 == 4:
                print('[%d. %5d] loss: %.3f' % (epoch + 1, i + 1, running_loss / 5))
                running_loss = 0.0

        # Serialization semantics, save the trained model
        torch.save(net.state_dict(), 'nn-audio-only-short-epoch-%d.pth' %(epoch))

        print('Finished Training')
        evaluate(net, testloader)

    print(mse_list)
    print(loss_emotion_epoch)

if __name__ == '__main__':
    main()
