import numpy as np
import matplotlib as mpl
mpl.use('Agg')
import matplotlib.pyplot as plt
import sys
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torch.utils.data
from torchvision import datasets, transforms
from torch.autograd import Variable
import torch.backends.cudnn as cudnn
from utilities import get_data, class_labels
#from utilities_nomfcc import get_data, class_labels

EPSIODE = 1000
batch_size = 32
device = 'cuda' if torch.cuda.is_available() else 'cpu'


class Flatten(nn.Module):
    def forward(self, x):
        x = x.view(x.size()[0], -1)
        return x

class CNN(nn.Module):
    def __init__(self, num_classes=4):
        super(CNN, self).__init__()
        self.conv1 = nn.Conv2d(1, 32, kernel_size=(13,1), stride=1, padding=0, bias=False)
        self.bn1 = nn.BatchNorm2d(32)
        self.conv2 = nn.Conv2d(32, 32, kernel_size=(13,1), stride=1, padding=0, bias=False)
        self.bn2 = nn.BatchNorm2d(32)
        self.conv3 = nn.Conv2d(32, 32, kernel_size=(13,1), stride=1, padding=0, bias=False)
        self.bn3 = nn.BatchNorm2d(32)
        self.conv4 = nn.Conv2d(32, 16, kernel_size=(7,1), stride=1, padding=0, bias=False)
        self.bn4 = nn.BatchNorm2d(16)
        self.conv5 = nn.Conv2d(16, 16, kernel_size=(7,1), stride=1, padding=0, bias=False)
        self.bn5 = nn.BatchNorm2d(16)
        self.conv6 = nn.Conv2d(16, 16, kernel_size=(7,1), stride=1, padding=0, bias=False)
        self.bn6 = nn.BatchNorm2d(16)
        self.flatten = Flatten()
        self.linear1 = nn.Linear(832,480)
        self.dropout1 = nn.Dropout(0.1)
        self.dropout = nn.Dropout(0.2)
        self.linear2 = nn.Linear(480, 240)
        self.linear3 = nn.Linear(240, num_classes)

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.dropout1(out)
        out = F.avg_pool2d(self.dropout1(F.relu(self.bn2(self.conv2(out)))), kernel_size=(2, 1))
        out = self.dropout1(F.relu(self.bn3(self.conv3(out))))
        out = F.avg_pool2d(self.dropout1(F.relu(self.bn4(self.conv4(out)))), kernel_size = (2,1))
        out = F.relu(self.bn5(self.conv5(out)))
        out = self.dropout1(out)
        out = self.dropout1(F.relu(self.bn6(self.conv6(out))))
        out1 = F.avg_pool2d(out, kernel_size=(2,1))
        out1 = self.flatten(out1)
        out2 = F.max_pool2d(out, kernel_size=(2,1))
        out2 = self.flatten(out2)
        out = torch.cat((out1, out2), 1)
        out = self.linear1(out)
        out = self.dropout1(out)
        out = self.dropout(self.linear2(out))
        out = F.log_softmax(self.linear3(out), dim=1)
        return out

model = CNN().float().to(device)
if device=='cuda':
    model = torch.nn.DataParallel(model)
    cudnn.benchmark = True

optimizer = optim.SGD(model.parameters(), lr=0.001)
#checkpoint = torch.load('../models/best_model_LSTM.h5')
#model.load_state_dict(checkpoint)

def train(epoch):
    global x_train, y_train
    for batch_idx, (data, target) in enumerate(train_loader):
        train_loss = 0
        data = data.view((len(data),1,320,1))  #(B, C, H, W): (32, 1, 320, 1)
        #print(data.shape) 
        data, target = data.to(device), target.to(device)
        optimizer.zero_grad()
        output = model(data)
        # loss
        #loss = F.binary_cross_entropy(output, target)
        loss = F.cross_entropy(output, (torch.max(target, 1)[1]))
        #print(loss)
        train_loss+=loss.item()
        loss.backward()
        # update
        optimizer.step()
        if batch_idx % 30 == 0:
            print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                epoch, batch_idx * len(data), len(x_train),
                100. * (batch_idx*len(data)) / len(x_train), loss.data[0]))
    return (train_loss/(batch_idx+1))
best_acc = 0
confusion_matrix_t = []
test_acc = []
def test():
    global x_test, y_test
    global best_acc
    model.eval()
    confusion_matrix = np.zeros((4,4))
    test_loss = 0
    correct = 0
    total = 0
    with torch.no_grad():
        for batch_idx, (data, target) in enumerate(test_loader):
            data = data.view((len(data),1,320,1))  #(B, C, H, W): (32, 1, 320, 1)
            data, target = data.to(device), target.to(device)
            output = model(data)
            #loss = F.binary_cross_entropy(output, target)
            loss = F.cross_entropy(output, (torch.max(target, 1)[1]))
            test_loss += loss.item()
            pred = output.data.max(1, keepdim=True)[1]
            total += target.size(0)
            # get the index of the max
            targ = target.data.max(1, keepdim=True)[1]
            for i, (p, t) in enumerate(zip(pred, targ)):
                if p == t:
                    correct += 1
                confusion_matrix[t-1][p-1]+=1
            #correct += pred.eq(target).sum().item()
            test_loss /= len(test_loader.dataset)
        acc = 100. * correct / len(test_loader.dataset)
        print('\nTest set: Average loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)\n'.format(
                test_loss, correct, len(test_loader.dataset),
                100. * correct / len(test_loader.dataset)))
        test_acc.append(acc)
        print("confusion matrix: ", confusion_matrix)
        if acc > best_acc:
            print('Saving..')
            best_acc = acc
            torch.save(model.state_dict(), best_model_path)
        confusion_matrix_t.append(confusion_matrix)

def to_categorical(y, num_classes):
    """ 1-hot encodes a tensor """
    return torch.tensor(np.eye(num_classes, dtype='int')[y])

if __name__ == "__main__":

    print('model given', 'CNN_without_mfcc')

    # Read data
    global x_train, y_train, x_test, y_test
    x_train, x_test, y_train, y_test = get_data(flatten=False)
    y_train = to_categorical(y_train, 4)
    y_test = to_categorical(y_test, 4)
    #print("y_train: ", y_train, type(y_train)) 

    torch_dataset = torch.utils.data.TensorDataset(x_train.float(), y_train)
    train_loader = torch.utils.data.DataLoader(dataset=torch_dataset,
                                           batch_size=batch_size,
                                           shuffle=True)
    torch_dataset_test = torch.utils.data.TensorDataset(x_test.float(), y_test)
    test_loader = torch.utils.data.DataLoader(dataset=torch_dataset_test,
                                          batch_size=batch_size,
                                          shuffle=False)

    global best_model_path
    best_model_path = '../models/best_model_CNN_nomfcc_nonsilence_SGD_50ms.h5'
    loss_l = []
    for epoch in range(1,EPSIODE):
        loss = train(epoch)
        test()
        loss_l.append(loss)

    print("best_acc: ", best_acc)
    print(confusion_matrix_t)
    print("loss_l: ", loss_l)
    print("test_acc: ", test_acc)
    plt.figure(1)
    plt.plot( np.arange(0, EPSIODE-1, 1), loss_l)
    plt.savefig('../results/CNN_nomfcc_nonsilence_20ms.png')
    plt.figure(2)
    plt.plot(np.arange(0, EPSIODE-1, 1), test_acc)
    plt.show()
    plt.savefig('../results/CNN_nomfcc_nonsilence_20ms_test_acc.png')
