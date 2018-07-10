import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torchvision
from liris_dataset import LirisDataset
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
        self.fc1 = nn.Linear(57344, 100)
        self.bn1 = nn.BatchNorm1d(100)
        self.fc2 = nn.Linear(100, 2)
        self.bn2 = nn.BatchNorm1d(2)

    def forward(self, x):
        x = self.bn1(F.relu(self.fc1(x)))
        x = self.bn2(F.relu(self.fc2(x)))

        return x

def evaluate(net, testloader):
    criterion = nn.MSELoss()
    loss_test = 0.0
    loss_emotion = [0.0] * 2
    for i, data in enumerate(testloader, 0):
        # get the inputs
        inputs = data['input']
        labels = data['labels']
        inputs, labels = inputs.to(device), labels.to(device)

        outputs = net(inputs)
        loss = criterion(outputs, labels)
        loss_test += loss.item()

        for i in range(2):
            loss = criterion(outputs[:, i], labels[:, i])
            loss_emotion[i] += loss.item()

    print('test result: ')
    for i in range(2):
        print('%s mse: %f' % (emotions[i], loss_emotion[i] / len(testloader)))
        loss_emotion_epoch[i].append(loss_emotion[i] / len(testloader))

    print('mse average: %f' % (loss_test / len(testloader)))
    mse_list.append(loss_test / len(testloader))

if __name__ == '__main__':
    batch_size = 8
    # Load and uniform DaLC Dataset
    trainset = liris_dataset.getLirisDataset('liris-accede-train-dataset.pkl', train=True, validate=False)
    validateset = liris_dataset.getLirisDataset('liris-accede-validate-dataset.pkl', train=True, validate=True)
    train_validateset = torch.utils.data.ConcatDataset([trainset, validateset])
    trainloader = torch.utils.data.DataLoader(train_validateset, batch_size=batch_size, shuffle=True)

    testset = liris_dataset.getLirisDataset('liris-accede-test-dataset.pkl', train=False, validate=False)
    testloader = torch.utils.data.DataLoader(testset, batch_size=batch_size, shuffle=False)
    
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
            labels = data['labels']
            inputs, labels = inputs.to(device), labels.to(device)

            # zero the parameter gradients
            optimizer.zero_grad()
            
            # a = list(netgparameters())[0].clone()
            # print(a.grad)

            # forward + backward + optimize
            outputs = net(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            # b = list(net.parameters())[0].clone()
 
            # print(torch.equal(a.data, b.data))

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
    print(loss_emotion_epoch)
