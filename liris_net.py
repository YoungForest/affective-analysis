import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torchvision
from liris_dataset import LirisDataset

class LirisNet(nn.Module):

    def __init__(self, input_dim):
        super(LirisNet, self).__init__()
        self.fc1 = nn.Linear(input_dim, 100)
        self.fc2 = nn.Linear(100, 2)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))

        return x

def evaluate(net, testloader):
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

    print('Test result: ')
    for i in range(2):
        print('%s MSE: %.3f' % (emotions[i], loss_emotion[i] / len(testloader)))
        loss_emotion_epoch[i].append(loss_emotion[i] / len(testloader))

    print('MSE average: %.3f' % (loss_test / len(testloader)))
    mse_list.append(loss_test / len(testloader))

# Training on GPU
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print(device)

mse_list = []
emotions = ['valence', 'arousal']
# record mse of two emotions every epoch
loss_emotion_epoch = [[], []]

if __name__ == '__main__':
    # Load and uniform DaLC Dataset
    trainset = LirisDataset(json_file='output-liris-resnet-34-kinetics.json', root_dir='/home/data_common/data_yangsen/data', transform=True, ranking_file='ACCEDEranking.txt', sets_file='ACCEDEsets.txt', sep='\t')
    trainloader = torch.utils.data.DataLoader(trainset, batch_size=32, shuffle=True)

    testset = LirisDataset(json_file='output-liris-resnet-34-kinetics.json', root_dir='/home/data_common/data_yangsen/data', transform=True, ranking_file='ACCEDEranking.txt', sets_file='ACCEDEsets.txt', sep='\t')
    testloader = torch.utils.data.DataLoader(testset, batch_size=4, shuffle=False)
    
    # new a Neural Network instance
    net = LirisNet(trainset.get_input_dim())
    if torch.cuda.device_count() > 1:
        print("Let's use", torch.cuda.device_count(), "GPUs!")
        net = nn.DataParallel(net)
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
            if i % 50 == 49:
                print('[%d. %5d] loss: %.3f' % (epoch + 1, i + 1, running_loss / 50))
                running_loss = 0.0

        print('Finished Training')
        evaluate(net, testloader)

    # Serialization semantics, save the trained model
    torch.save(net.state_dict(), 'nn-1.pth')

    print(mse_list)
    print(emotions_mse_list)
