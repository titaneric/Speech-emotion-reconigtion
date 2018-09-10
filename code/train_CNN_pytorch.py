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
from torchvision import datasets, transforms
from torch.autograd import Variable
import torch.backends.cudnn as cudnn
from utilities import get_data, class_labels

models = ["CNN", "LSTM"]
batch_size = 32
EPISODE = 100
device = 'cuda' if torch.cuda.is_available() else 'cpu'

class Flatten(nn.Module):
    def forward(self, x):
        x = x.view(x.size()[0], -1)
        return x

class CNN(nn.Module):
    def __init__(self, num_classes=4):
        super(CNN, self).__init__()
        self.conv1 = nn.Conv2d(1, 8, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(8)
        self.conv2 = nn.Conv2d(8, 8, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(8)
        self.conv3 = nn.Conv2d(8, 8, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn3 = nn.BatchNorm2d(8)
        self.conv4 = nn.Conv2d(8, 16, kernel_size=2, stride=1, padding=1, bias=False)
        self.bn4 = nn.BatchNorm2d(16)
        self.flatten = Flatten()
        self.linear1 = nn.Linear(640,16)
        self.bn5 = nn.BatchNorm1d(16)
        self.dropout = nn.Dropout(0.2)
        self.linear2 = nn.Linear(16, num_classes)

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = F.max_pool2d(F.relu(self.bn2(self.conv2(out))), kernel_size=(2, 1))
        out = F.relu(self.bn3(self.conv3(out)))
        out = self.bn4(self.conv4(out))
        out = F.relu(out)
        out = F.max_pool2d(out, kernel_size=(2,1))
        out = self.flatten(out)
        out = self.linear1(out)
        out = F.relu(self.bn5(out))
        out = self.dropout(out)
        out = F.log_softmax(self.linear2(out), dim=1)
        return out

model = CNN().double().to(device)
if device=='cuda':
    model = torch.nn.DataParallel(model)
    cudnn.benchmark = True

optimizer = optim.Adam(model.parameters(), lr=0.001)
#checkpoint = torch.load('../models/best_model_CNN_chunk.h5')
#model.load_state_dict(checkpoint)

def train(epoch):
    global x_train, y_train
    train_loss = 0
    for batch_idx, (data, target) in enumerate(train_loader):
        data = data.view((len(data),1 , 3,39))  #(B, C, H, W): (32, 1, 3744, 39)
        #print(data.shape) 
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
                100. * (batch_idx*len(data)) / len(x_train), loss.data[0]))
    return (train_loss/(batch_idx+1))
best_acc = 0
test_acc = []
confusion_matrix_t = []
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
            data = data.view((len(data),1, 3, 39))  #(B, C, H, W): (32, 1, 3744, 39)
            data, target = data.to(device), target.to(device)
            output = model(data)
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
            
        test_loss /= len(test_loader.dataset)
        acc = 100. * correct / len(test_loader.dataset)
        print('\nTest set: Average loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)\n'.format(
                test_loss, correct, len(test_loader.dataset),
                100. * correct / len(test_loader.dataset)))
        print("confusion matrix: ", confusion_matrix)
        test_acc.append(acc)
        confusion_matrix_t.append(confusion_matrix)
        if acc > best_acc:
            print('Saving..')
            best_acc = acc
            torch.save(model.state_dict(), best_model_path)


def to_categorical(y, num_classes):
    """ 1-hot encodes a tensor """
    return torch.tensor(np.eye(num_classes, dtype='int')[y])

loss_l = []
if __name__ == "__main__":

    if len(sys.argv) != 2:
        sys.stderr.write('Invalid arguments\n')
        sys.stderr.write('Usage python2 train_DNN.py <model_number>\n')
        sys.stderr.write('1 - CNN\n')
        sys.stderr.write('2 - LSTM\n')
        sys.exit(-1)

    n = int(sys.argv[1]) - 1
    print('model given', models[n])

    # Read data
    global x_train, y_train, x_test, y_test
    x_train, x_test, y_train, y_test = get_data(flatten=False)
    y_train = to_categorical(y_train, 4)
    y_test = to_categorical(y_test, 4)

    if n == 0:
        torch_dataset = torch.utils.data.TensorDataset(x_train, y_train)
        train_loader = torch.utils.data.DataLoader(dataset=torch_dataset,
                                           batch_size=batch_size,
                                           shuffle=True)
        torch_dataset_test = torch.utils.data.TensorDataset(x_test, y_test)
        test_loader = torch.utils.data.DataLoader(dataset=torch_dataset_test,
                                          batch_size=batch_size,
                                          shuffle=False)
    elif n > len(models):
        sys.stderr.write('Model Not Implemented yet')
        sys.exit(-1)

    global best_model_path
    best_model_path = '../models/best_model_CNN_mfcc_50mc_chunk.h5'

    for epoch in range(1,EPISODE):
        loss = train(epoch)
        test()
        loss_l.append(loss)
    print("best_acc: ", best_acc)
    print(confusion_matrix_t)
    print("loss_l: ", loss_l)
    print("test_acc: ", test_acc)
    plt.figure(1)
    plt.plot( np.arange(0, EPISODE-1, 1), loss_l)
    plt.savefig('../results/CNN_mfcc_nonsilence_SGD_50ms.png')
    plt.figure(2)
    plt.plot(np.arange(0, EPISODE-1, 1), test_acc)
    plt.show()
    plt.savefig('../results/CNN_mfcc_nonsilence_SGD_50ms_test_acc.png')

