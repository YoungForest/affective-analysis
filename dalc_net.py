import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torchvision
from dalc_dataset import DalcDataset

class DalcNet(nn.Module):

    def __init__(self, input_dim):
        super(DalcNet, self).__init__()
        self.fc1 = nn.Linear(input_dim, 100)
        self.fc2 = nn.Linear(100, 8)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))

        return x

def evaluate(net, testloader):
    loss_test = 0.0
    loss_emotion = [0.0] * 8
    for i, data in enumerate(testloader, 0):
        # get the inputs
        inputs = data['input']
        labels = data['labels']
        inputs, labels = inputs.to(device), labels.to(device)

        outputs = net(inputs)
        loss = criterion(outputs, labels)
        loss_test += loss.item()

        for i in range(8):
            loss = criterion(outputs[:, i], labels[:, i])
            loss_emotion[i] += loss.item()

    print('Test result: ')
    emotions = ['arousal', 'excitement', 'pleasure', 'contentment', 'sleepiness', 'depression', 'misery', 'distress']
    for i in range(8):
        print('%s MSE: %.3f' % (emotions[i], loss_emotion[i] / len(testloader)))
        emotions_mse_list[i].append(loss_emotion[i] / len(testloader))

    print('MSE average: %.3f' % (loss_test / len(testloader)))
    mse_list.append(loss_test / len(testloader))

# Training on GPU
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print(device)

mse_list = []
emotions_mse_list = [[], [], [], [], [], [], [], []]
if __name__ == '__main__':
    # Load and uniform DaLC Dataset
    trainset = DalcDataset('output-resnet-34-kinetics.json', '/home/data_common/data_yangsen/videos', train=True, transform=True, ranking_file='filled-labels_features.csv', sets_file='cv_id_10.txt', sep=',')
    trainloader = torch.utils.data.DataLoader(trainset, batch_size=4, shuffle=True)

    testset = DalcDataset('output-resnet-34-kinetics.json', '/home/data_common/data_yangsen/videos', train=False, transform=True, ranking_file='filled-labels_features.csv', sets_file='cv_id_10.txt', sep=',')
    testloader = torch.utils.data.DataLoader(testset, batch_size=4, shuffle=False, num_workers=1)
    
    # new a Neural Network instance
    net = DalcNet(trainset.get_input_dim())
    net.to(device)
 
    # define a Loss function and optimizer
    criterion = nn.MSELoss()
    optimizer = optim.SGD(net.parameters(), lr=0.001, momentum=0.9)


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

            # forward + backward + optimize
            outputs = net(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            # print statistics
            running_loss += loss.item()
            if i % 5 == 4:
                print('[%d. %5d] loss: %.3f' % (epoch + 1, i + 1, running_loss / 5))
                running_loss = 0.0

        print('Finished Training')
        evaluate(net, testloader)

    print(mse_list)
    print(emotions_mse_list)
