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

from convlstm import ConvLSTM

models = ["CNN", "LSTM"]
batch_size = 32
EPISODE = 100
device = 'cuda' if torch.cuda.is_available() else 'cpu'
channels = 1
height, width = (198, 39)
class MyConvLSTM(nn.Module):
    def __init__(self, num_classes=4):
        super(MyConvLSTM, self).__init__()
        self.convLSTM = ConvLSTM(input_size=(height, width),
                        input_dim=channels,
                        hidden_dim=[64],
                        kernel_size=(3, 3),
                        num_layers=1,
                        batch_first=True,
                        bias=True,
                        return_all_layers=False)

        self.linear = nn.Linear(64, num_classes)
    def forward(self, x):
        # x = torch.cat((x, x))
        x = x.view((-1, 1, 1, 198, 39)).float()  #(B, T, C, H, W): (3, 32, 1, 3744, 39)
        r_out_list, r_state_list = self.convLSTM(x, None)
        r_out = r_out_list[0]

        r_out = r_out.view((len(r_out), -1, 64))
        r_out = self.linear(r_out[:,-1,:])
        return r_out




model = MyConvLSTM()
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
                100. * (batch_idx*len(data)) / len(x_train), loss.item()))
    return (train_loss/(batch_idx+1))
best_acc = 0
test_acc = []
confusion_matrix_max = np.zeros((4,4))
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
            output = model(data)
            # print(output.type())
            # print(torch.max(target, 1)[1].cuda().float().type())
            tmp_target = torch.max(target, 1)[1].cuda().long()

            loss = F.cross_entropy(output, tmp_target)
            test_loss += loss.item()
            pred = output.data.max(1, keepdim=True)[1]
            total += target.size(0)
            # get the index of the max
            targ = target.data.max(1, keepdim=True)[1]
            for i, (p, t) in enumerate(zip(pred, targ)):
                if p == t.cuda():
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


def to_categorical(y, num_classes):
    """ 1-hot encodes a tensor """
    return torch.tensor(np.eye(num_classes, dtype='int')[y])

loss_l = []
if __name__ == "__main__":

    # if len(sys.argv) != 2:
    #     sys.stderr.write('Invalid arguments\n')
    #     sys.stderr.write('Usage python2 train_DNN.py <model_number>\n')
    #     sys.stderr.write('1 - CNN\n')
    #     sys.stderr.write('2 - LSTM\n')
    #     sys.exit(-1)

    # n = int(sys.argv[1]) - 1
    # print('model given', models[n])

    # Read data
    global x_train, y_train, x_test, y_test
    x_train, x_test, y_train, y_test = get_data(flatten=False)
    y_train = to_categorical(y_train, 4)
    y_test = to_categorical(y_test, 4)

    # if n == 0:
    torch_dataset = torch.utils.data.TensorDataset(x_train, y_train)
    train_loader = torch.utils.data.DataLoader(dataset=torch_dataset,
                                        batch_size=batch_size,
                                        shuffle=True)
    torch_dataset_test = torch.utils.data.TensorDataset(x_test, y_test)
    test_loader = torch.utils.data.DataLoader(dataset=torch_dataset_test,
                                        batch_size=batch_size,
                                        shuffle=False)
    # elif n > len(models):
    #     sys.stderr.write('Model Not Implemented yet')
    #     sys.exit(-1)

    global best_model_path
    best_model_path = '../models/best_model_CNN_mfcc_50mc_chunk.h5'

    for epoch in range(1,EPISODE):
        loss = train(epoch)
        test()
        loss_l.append(loss)
    print("best_acc: ", best_acc)
    print("confusion_matrix_max: ", confusion_matrix_max)
    print("loss_l: ", loss_l)
    print("test_acc: ", test_acc)
    plt.figure(1)
    plt.plot( np.arange(0, EPISODE-1, 1), loss_l)
    plt.savefig('../results/CNN_mfcc_nonsilence_SGD_50ms.png')
    plt.figure(2)
    plt.plot(np.arange(0, EPISODE-1, 1), test_acc)
    plt.show()
    plt.savefig('../results/CNN_mfcc_nonsilence_SGD_50ms_test_acc.png')

