from torchvision import models
import torch.nn as nn
import torch
import torch.nn.functional as F
import torch.optim as optim
from su_dataset import SuDataset
import pickle
import os
import numpy as np
from torch.utils.data.sampler import SubsetRandomSampler



# Training on GPU
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
# device = torch.device("cpu") # using CPU only
print(device)

mse_list = []
emotions = ['arousal', 'excitement', 'pleasure', 'contentment', 'sleepiness', 'depression', 'misery', 'distress']
# record mse of two emotions every epoch
loss_emotion_epoch = [[], [], [], [], [], [], [], []]

def evaluate(net, testloader):
    criterion = nn.MSELoss()
    loss_test = 0.0
    loss_emotion = [0.0] * 8
    for i, data in enumerate(testloader, 0):
        # get the inputs
        inputs = data['mfcc']
        labels = data['labels']
        inputs.unsqueeze_(1)
        inputs, labels = inputs.to(device), labels.to(device)

        outputs = net(inputs)[0]
        loss = criterion(outputs, labels)
        loss_test += loss.item()

        for i in range(8):
            loss = criterion(outputs[:, i], labels[:, i])
            loss_emotion[i] += loss.item()

    print('test result: ')
    for i in range(8):
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
        self.fc1 = nn.Linear(32 * 10 * 56, 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, 8)

    def forward(self, x):
        # Max poolint over a (2, 2) window
        # print ('73 ', x.shape)
        x = self.bn0(x)
        # print ('75 ', x.shape)
        x = F.max_pool2d(F.relu(self.bn1(self.conv1(x))), 2)
        # print ('78 ', x.shape)
        x = F.max_pool2d(F.relu(self.bn2(self.conv2(x))), (1, 2))
        # print ('79 ', x.shape)
        x = F.max_pool2d(F.relu(self.bn3(self.conv3(x))), (1, 2))
        # print ('81 ', x.shape)
        # x = F.max_pool2d(F.relu(self.conv1(x)), 2)
        # x = F.max_pool2d(F.relu(self.conv2(x)), 2)
        # x = F.max_pool2d(F.relu(self.conv3(x)), 2)
        features = x.view(-1, self.num_flat_features(x))
        # print ('86 ', features.shape)
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
    cache = 'cache.pkl'
    dataset = None
    if os.path.exists(cache):
        with open(cache, 'rb') as my_file:
            dataset = pickle.load(my_file)
    else:
        dataset = SuDataset('../audios', 'emotion.txt')
        with open(cache, 'wb') as out:
            pickle.dump(dataset, out, pickle.HIGHEST_PROTOCOL)
    batch_size = 16
    validation_split = .2
    shuffle_dataset = True
    random_seed= 42

    # Creating data indices for training and validation splits:
    dataset_size = len(dataset)
    indices = list(range(dataset_size))
    split = int(np.floor(validation_split * dataset_size))
    if shuffle_dataset :
        np.random.seed(random_seed)
        np.random.shuffle(indices)
    train_indices, val_indices = indices[split:], indices[:split]

    # Creating PT data samplers and loaders:
    train_sampler = SubsetRandomSampler(train_indices)
    validation_sampler = SubsetRandomSampler(val_indices)

    train_loader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, 
                                                       sampler=train_sampler)
    testloader = torch.utils.data.DataLoader(dataset, batch_size=batch_size,
                                                       sampler=validation_sampler)
    net = AudioNet()
    if torch.cuda.device_count() > 1:
        print("Let's use", torch.cuda.device_count(), "GPUs!")
        net = nn.DataParallel(net)
    net.to(device)
    # net.load_state_dict(torch.load('nn-epoch-49.pth'))
 
    # define a Loss function and optimizer
    criterion = nn.MSELoss()
    optimizer = optim.SGD(net.parameters(), lr=0.0001)


    # train the network
    for epoch in range(50): # Loop over the dataset multiple times

        running_loss = 0.0
        for i, data in enumerate(train_loader, 0):
            # get the inputs
            inputs = data['mfcc']
            labels = data['labels']
            inputs.unsqueeze_(1)
            inputs, labels = inputs.to(device), labels.to(device)

            # zero the parameter gradients
            optimizer.zero_grad()

            # forward + backward + optimize
            # print ('inputs shape', inputs.shape)
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
