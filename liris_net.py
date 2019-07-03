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

batch_size = 128

# Training on GPU
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print(device)
date = '7_3'
writer = SummaryWriter('log/')

mse_list = []
emotions = ['valence', 'arousal']
# record mse of two emotions every epoch
loss_emotion_epoch = [[], []]


class LirisNet(nn.Module):

    def __init__(self):
        super(LirisNet, self).__init__()
        self.input_dim = 14336
        self.hidden_dim = 64
        self.num_directions = 2
        self.num_layers = 2
        self.batch_size = batch_size
        self.lstm = nn.LSTM(input_size=self.input_dim, hidden_size=self.hidden_dim,
                            num_layers=self.num_layers, bidirectional=(self.num_directions == 2))
        self.linear = nn.Linear(
            self.num_directions*self.hidden_dim*self.batch_size, 2 * self.batch_size)

    def forward(self, x):
        a, b = x.shape
        x = x.view(1, a, b)
        # lstm input: (seq_len, batch, input_size), (1, 16, 14336)
        lstm_out, _ = self.lstm(x)
        y_pred = self.linear(lstm_out[-1].reshape(1, -1).squeeze())

        return y_pred.reshape(self.batch_size, 2)

evaluate_count = 0
def evaluate(net, testloader):
    criterion = nn.MSELoss()
    loss_test = 0.0
    for i, data in enumerate(testloader, 0):
        # get the inputs
        inputs = data['input']
        valence = data['labels'][:, 0:1]
        arousal = data['labels'][:, 1:2]
        inputs, valence, arousal = inputs.to(
            device), valence.to(device), arousal.to(device)
        if inputs.shape[0] != batch_size:
            print(f'bad shape when test: {inputs.shape}')
            continue
        outputs = net(inputs)
        loss = criterion(outputs, arousal)
        loss_test += loss.item()

    print('mse average: %f' % (loss_test / len(testloader)))
    global evaluate_count
    writer.add_scalar(f'test_average_{date}', loss_test / len(testloader), evaluate_count)
    evaluate_count += 1
    mse_list.append(loss_test / len(testloader))


if __name__ == '__main__':
    train_dataset = LirisDataset(json_file='output-liris-resnet-34-kinetics.json', root_dir=movies.data_path,
                                 transform=True, window_size=3, ranking_file=movies.ranking_file, sets_file=movies.sets_file, sep='\t')
    split_point = int(len(train_dataset) * 3 / 4)
    trainloader = torch.utils.data.DataLoader(
        torch.utils.data.Subset(train_dataset, range(0, split_point)), batch_size=batch_size, shuffle=False)
    testloader = torch.utils.data.DataLoader(
        torch.utils.data.Subset(train_dataset, range(split_point, len(train_dataset))), batch_size=batch_size, shuffle=False)

    # new a Neural Network instance
    net = LirisNet().cuda()
    if torch.cuda.device_count() > 1:
        print("Let's use", torch.cuda.device_count(), "GPUs!")
        net = nn.DataParallel(net)
    # net.load_state_dict(torch.load(
    #     '/data/pth/nn-6_28-epoch-49.pth'))

    # define a Loss function and optimizer
    criterion = nn.MSELoss()
    optimizer = optim.SGD(net.parameters(), lr=0.0001)
    # print(net.parameters())

    # train the network
    epoch_num = 150
    count = 0
    for epoch in range(epoch_num):  # Loop over the dataset multiple times
        for i, data in enumerate(trainloader, 0):
            # get the inputs
            inputs = data['input']
            valence = data['labels'][:, 0:1]
            arousal = data['labels'][:, 1:2]
            inputs, valence, arousal = inputs.to(
                device), valence.to(device), arousal.to(device)
            if inputs.shape[0] != batch_size:
                print(f'bad shape: {inputs.shape}')
                continue
            # zero the parameter gradients
            optimizer.zero_grad()

            # forward + backward + optimize
            outputs = net(inputs)
            loss = criterion(outputs, arousal)
            loss.backward()
            optimizer.step()

            # print statistics
            writer.add_scalar(f'train_{date}', loss.item(), count)
            count += 1
            print(f'Epoch: {evaluate_count}, index: {i}, count: {count}: {loss.item()}')
        # Serialization semantics, save the trained model
        torch.save(net.state_dict(
        ), f'/data/pth/nn-{date}-epoch-{evaluate_count}.pth')

        print('Finished Training')
        evaluate(net, testloader)

    print(mse_list)
