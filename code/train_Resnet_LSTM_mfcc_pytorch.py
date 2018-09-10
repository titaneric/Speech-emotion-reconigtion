##############################################
# Resnet20+ LSTM with additional dropout and relu output layers
# lr=0.001
# equal distribution sampling 
# LSTM layers = 1, softmax
##############################################
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import sys
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torch.utils.data
import torch.autograd as autograd
from torchvision import datasets, transforms
from torch.autograd import Variable
import torch.backends.cudnn as cudnn
from utilities import get_data, class_labels

batch_size = 32
EPSIODE = 5000
device = 'cuda' if torch.cuda.is_available() else 'cpu'

class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, in_planes, planes, stride=1):
        super(BasicBlock, self).__init__()
        self.conv1 = nn.Conv2d(in_planes, planes, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.dropout = nn.Dropout(0.2)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)
        self.shortcut = nn.Sequential()
        if stride != 1 or in_planes != self.expansion*planes:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_planes, self.expansion*planes, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(self.expansion*planes)
            )

    def forward(self, x):
        out = self.dropout(F.relu(self.bn1(self.conv1(x))))
        out = self.bn2(self.conv2(out))
        out += self.shortcut(x)
        out = F.relu(out)
        return out


class ResNet_LSTM(nn.Module):
    def __init__(self, block, num_blocks, num_classes=4):
        super(ResNet_LSTM, self).__init__()
        self.in_planes = 8
        self.conv1 = nn.Conv2d(1, 8, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(8)
        self.layer1 = self._make_layer(block, 8, num_blocks[0], stride=1)
        self.layer2 = self._make_layer(block, 16, num_blocks[1], stride=2)
        self.layer3 = self._make_layer(block, 32, num_blocks[2], stride=2)
        self.rnn = nn.LSTM(         # if use nn.RNN(), it hardly learns
            input_size=640,
            hidden_size=64,         # rnn hidden unit
            num_layers=1,           # number of rnn layer
            batch_first=True,       # input & output will has batch size as 1s dimension. e.g. (batch, time_step, input_size)
        )
        self.dropout = nn.Dropout(0.2) 
        self.linear = nn.Linear(64, 16)
        self.linear2 = nn.Linear(16, num_classes)

    def _make_layer(self, block, planes, num_blocks, stride):
        strides = [stride] + [1]*(num_blocks-1)
        layers = []
        for stride in strides:
            layers.append(block(self.in_planes, planes, stride))
            self.in_planes = planes * block.expansion
        return nn.Sequential(*layers)

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.layer1(out)
        out = self.layer2(out)
        out = self.layer3(out)
        out = out.view((out.size(0), 25, -1))
        r_out, (h_n, h_c) = self.rnn(out, None)
        r_out = r_out[:, -1, :]
        """
        add two ReLU layers
        and one dropout layer
        and one linear layer
        """
        out = F.relu(self.linear(r_out))
        out = self.dropout(out)
        out = F.relu(self.linear2(out))
        out = F.softmax(out, dim=0)
        return out


model = ResNet_LSTM(BasicBlock, [3,3,3]).double().to(device)
if device=='cuda':
    model = torch.nn.DataParallel(model)
    cudnn.benchmark = True

optimizer = optim.Adam(model.parameters(), lr=0.001)


#print("Loading...")
#checkpoint = torch.load('../models/best_model_LSTM.h5')
#model.load_state_dict(checkpoint)

def train(epoch):
    model.train()
    global x_train, y_train
    train_loss = 0
    for batch_idx, (data, target) in enumerate(train_loader):
        data = data.view((len(data),1,198,39)).double() #(B, C, H, W): (32, 1, 198, 39)
        data, target = data.to(device), target.to(device)
        optimizer.zero_grad()
        output = model(data)
        # loss
        loss = F.cross_entropy(output, (torch.max(target, 1)[1]))
        train_loss+=loss.item()
        loss.backward()
        # update
        optimizer.step()
        if batch_idx % 10 == 0:
            print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                epoch, batch_idx * len(data), len(x_train),
                100. * batch_idx / len(x_train), loss.data[0]))
        return (train_loss/(batch_idx+1))

confusion_matrix_max = np.zeros((4,4))
best_acc = 0
test_acc = []

def test():
    global x_test, y_test
    global best_acc
    global confusion_matrix_max
    model.eval()
    confusion_matrix = np.zeros((4,4))
    test_loss = 0
    correct = 0
    total = 0
    with torch.no_grad():
        for batch_idx, (data, target) in enumerate(test_loader):
            data = data.view((len(data),1,198,39)).double()
            data, target = data.to(device), target.to(device)
            output = model(data)
            #print("output of test: ", output, output.size())
            loss = F.cross_entropy(output, (torch.max(target, 1)[1]))
            test_loss += loss.item()
            pred = output.data.max(1, keepdim=True)[1]
            total += target.size(0)
            # get the index of the max
            targ = target.data.max(1, keepdim=True)[1]
            for i, (p, t) in enumerate(zip(pred, targ)):
                if p == t:
                    correct += 1
                confusion_matrix[t][p]+=1
            test_loss /= len(test_loader.dataset)
        acc = 100. * correct / len(test_loader.dataset)
        print('\nTest set: Average loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)\n'.format(
                test_loss, correct, len(test_loader.dataset),
                100. * correct / len(test_loader.dataset)))
        print("confusion matrix: ", confusion_matrix)
        test_acc.append(acc)
        if acc > best_acc:
            print('Saving..')
            best_acc = acc
            torch.save(model.state_dict(), best_model_path)
            confusion_matrix_max = confusion_matrix
            print(confusion_matrix_max)


def to_categorical(y, num_classes):
    """ 1-hot encodes a tensor """
    return torch.tensor(np.eye(num_classes, dtype='int')[y])

if __name__ == "__main__":

    print('model given', "Resnet_LSTM")

    # Read data
    global x_train, y_train, x_test, y_test
    x_train, x_test, y_train, y_test = get_data(flatten=False)
    y_train = to_categorical(y_train, 4)
    y_test = to_categorical(y_test, 4)

    torch_dataset = torch.utils.data.TensorDataset(x_train, y_train)
    train_loader = torch.utils.data.DataLoader(dataset=torch_dataset,
                                           batch_size=batch_size,
                                           shuffle=True)
    torch_dataset_test = torch.utils.data.TensorDataset(x_test, y_test)
    test_loader = torch.utils.data.DataLoader(dataset=torch_dataset_test,
                                          batch_size=batch_size,
                                          shuffle=False)
    
    global best_model_path
    best_model_path = '../models/best_model_LSTM_Resnet.h5'
    loss_l = []
    for epoch in range(1,EPSIODE):
        loss = train(epoch)
        test()
        loss_l.append(loss)
        if epoch % 1000 ==0:
            plt.figure(1)
            plt.plot( np.arange(0, epoch, 1), loss_l)
            plt.savefig('../results/Resnet_LSTM_3000.png')
            plt.figure(2)
            plt.plot(np.arange(0, epoch, 1), test_acc)
            plt.show()
            plt.savefig('../results/Resnet_LSTM_3000_test_acc.png')
        
    print("best_acc: ", best_acc)
    print(confusion_matrix_max)
    print("loss_l: ", loss_l)
    print("test_acc: ", test_acc)
    plt.figure(1)
    plt.plot( np.arange(0, EPSIODE-1, 1), loss_l)
    plt.savefig('../results/Resnet_LSTM_3000.png')
    plt.figure(2)
    plt.plot(np.arange(0, EPSIODE-1, 1), test_acc)
    plt.show()
    plt.savefig('../results/Resnet_LSTM_3000_test_acc.png')


