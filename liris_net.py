import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torchvision
from liris_dataset import LirisDataset
from liris_dataset import getDataLoader
import liris_dataset
from torch.utils.data.sampler import SubsetRandomSampler
import movies
from tensorboardX import SummaryWriter
import pandas as pd

batch_size = 4  # squence length

# Training on GPU
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print(device)
date = '7_29'
writer = SummaryWriter('log/')

mse_list = []
emotions = ['valence', 'arousal']
# record mse of two emotions every epoch
loss_emotion_epoch = [[], []]


class LirisNet(nn.Module):

    def __init__(self):
        super(LirisNet, self).__init__()
        self.input_dim = 12288
        self.hidden_dim = 2048
        self.num_directions = 2
        self.num_layers = 2
        self.batch_size = batch_size
        self.bn1 = nn.BatchNorm1d(self.input_dim)
        self.lstm = nn.LSTM(input_size=self.input_dim, hidden_size=self.hidden_dim,
                            num_layers=self.num_layers, bidirectional=(self.num_directions == 2))
        self.bn2 = nn.BatchNorm1d(self.num_directions*self.hidden_dim)
        self.linear = nn.Linear(
            self.num_directions*self.hidden_dim, 2)

    def forward(self, x):
        a, b = x.shape  # seqence length, input dim
        assert(a == self.batch_size)
        x = self.bn1(x)
        x = x.view(a, 1, b)
        # lstm input: (seq_len, batch, input_size)
        # output: seq_len, batch, num_directions * hidden_size
        lstm_out, _ = self.lstm(x)
        lstm_out = lstm_out.view(a, self.num_directions*self.hidden_dim)
        # lstm_out = self.bn2(lstm_out)
        y_pred = self.linear(lstm_out)

        return y_pred


evaluate_count = 0
result = []

def evaluate(net, testloader, test=True, label='test'):
    criterion = nn.MSELoss()
    arousal_loss_test = 0.0
    valence_loss_test = 0.0
    for i, data in enumerate(testloader, 0):
        # get the inputs
        inputs = data['input']
        valence = data['labels'][:, 0:1]
        arousal = data['labels'][:, 1:2]
        ground_truth = data['labels']
        inputs, valence, arousal, ground_truth = inputs.to(
            device), valence.to(device), arousal.to(device), ground_truth.to(device)
        if inputs.shape[0] != batch_size:
            print(f'bad shape when test: {inputs.shape}')
            continue
        outputs = net(inputs)
        # print(outputs)
        # print(i)
        # print(data)
        for i, item in enumerate(data['video']):
            row = [item, valence[i].item(), arousal[i].item(), outputs[i]
                   [0].item(), outputs[i][1].item(), test]
            result.append(row)
        outputs *= 1
        valence_loss = criterion(outputs[:, 0:1], valence)
        arousal_loss = criterion(outputs[:, 1:2], arousal)
        arousal_loss_test += arousal_loss.item()
        valence_loss_test += valence_loss.item()

    print(f'{label} arousal mse average: {arousal_loss_test / len(testloader)}')
    print(f'{label} valence mse average: {valence_loss_test / len(testloader)}')
    writer.add_scalar(f'predict_arousal_{label}_average_{date}',
                      arousal_loss_test / len(testloader), evaluate_count)
    writer.add_scalar(f'predict_valence_{label}_average_{date}',
                      valence_loss_test / len(testloader), evaluate_count)
    mse_list.append((arousal_loss_test / len(testloader), valence_loss_test / len(testloader)))

def predict_all():
    train_dataset = LirisDataset(json_file='resnext-101-output.json', root_dir=movies.data_path,
                                 transform=True, window_size=1, ranking_file=movies.ranking_file, sets_file=movies.sets_file, sep='\t')
    split_point = int(len(train_dataset) * 3 / 4)
    totalloader = torch.utils.data.DataLoader(
        train_dataset, batch_size=batch_size, shuffle=False)
    trainloader = torch.utils.data.DataLoader(
        torch.utils.data.Subset(train_dataset, range(0, split_point)), batch_size=batch_size, shuffle=False)
    testloader = torch.utils.data.DataLoader(
        torch.utils.data.Subset(train_dataset, range(split_point, len(train_dataset))), batch_size=batch_size, shuffle=False)
    net = LirisNet().cuda()
    net.load_state_dict(torch.load(
        '/data/pth/nn-7_16-epoch-99.pth'))
    evaluate(net, trainloader, False)
    evaluate(net, testloader, True)
    predict_result = pd.DataFrame(data=result, columns=[
                                  'name', 'ground_truth_valence', 'ground_truth_arousal', 'valence', 'arousal', 'test'])
    predict_result.to_csv('predict.csv')

if __name__ == '__main__':
    train_dataset = LirisDataset(json_file='resnext-101-output.json', root_dir=movies.data_path,
                                 transform=True, window_size=1, ranking_file=movies.ranking_file, sets_file=movies.sets_file, sep='\t')
    train_split_point = int(len(train_dataset) * 1 / 2)
    valid_split_point = int(len(train_dataset) * 3 / 4)
    trainloader = torch.utils.data.DataLoader(
        torch.utils.data.Subset(train_dataset, range(0, train_split_point)), batch_size=batch_size, shuffle=False)
    testloader = torch.utils.data.DataLoader(
        torch.utils.data.Subset(train_dataset, range(train_split_point, len(train_dataset))), batch_size=batch_size, shuffle=False)

    # new a Neural Network instance
    net = LirisNet().cuda()
    if torch.cuda.device_count() > 1:
        print("Let's use", torch.cuda.device_count(), "GPUs!")
        net = nn.DataParallel(net)
    # net.load_state_dict(torch.load(
    #     '/data/pth/nn-7_18-epoch-199.pth'))

    # # define a Loss function and optimizer
    criterion = nn.MSELoss()
    optimizer = optim.SGD(net.parameters(), lr=0.0001)
    # print(net.parameters())

    # train the network
    epoch_num = 500
    count = 0
    for epoch in range(epoch_num):  # Loop over the dataset multiple times
        train_loss = 0.0
        net.train()
        for i, data in enumerate(trainloader, 0):
            # get the inputs
            inputs = torch.cat((data['input'][:, ::2][:, :6144], data['audio'][:, ::2][:, :6144]), 1)
            valence = data['labels'][:, 0:1]
            arousal = data['labels'][:, 1:2]
            ground_truth = data['labels']
            inputs, valence, arousal, ground_truth = inputs.to(
                device), valence.to(device), arousal.to(device), ground_truth.to(device)
            if inputs.shape[0] != batch_size:
                print(f'bad shape: {inputs.shape}')
                continue
            # zero the parameter gradients
            optimizer.zero_grad()

            # forward + backward + optimize
            inputs = inputs.requires_grad_()
            outputs = net(inputs)
            loss = criterion(outputs, ground_truth)
            loss.backward()
            optimizer.step()

            # print statistics
            train_loss += loss.item()
            
        writer.add_scalar(f'train_loss_{date}', train_loss / len(trainloader), epoch)
        print(f'Epoch: {epoch}: {train_loss / len(trainloader)}')
        # Serialization semantics, save the trained model
        torch.save(net.state_dict(
        ), f'/data/pth/nn-{date}-epoch-{evaluate_count}.pth')

        print('Finished Training')
        net.eval()
        evaluate(net, testloader, label='test')
        evaluate_count += 1
    
    print(mse_list)
