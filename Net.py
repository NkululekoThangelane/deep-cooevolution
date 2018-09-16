import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.multiprocessing as mp
import math
import numpy as np
class Net(nn.Module):
    def __init__(self, init=False):
        super(Net, self).__init__()
        self.conv1 = nn.Conv2d(1, 10, kernel_size=5)
        self.conv2 = nn.Conv2d(10, 20, kernel_size=5)
        #self.conv2_drop = nn.Dropout2d()
        self.fc1 = nn.Linear(320, 50)
        self.fc2 = nn.Linear(50, 10)


        if init==True:

            self.initial_wieghts()



    def forward(self, x):
        x = F.relu(F.max_pool2d(self.conv1(x), 2))
        x = F.relu(F.max_pool2d(self.conv2(x), 2))
        x = x.view(-1, 320)
        x = F.relu(self.fc1(x))
        #x = F.dropout(x, training=self.training)
        x = self.fc2(x)
        return F.log_softmax(x,dim=1)

    def extract_parameters(self):
        tot_size = self.count_parameters()
        pvec = np.zeros(tot_size, np.float32)
        count = 0
        for param in self.parameters():
            sz = param.cpu().data.numpy().flatten().shape[0]
            pvec[count:count + sz] = param.cpu().data.numpy().flatten()
            count += sz
        return pvec.copy()

    def inject_parameters(self, pvec):
        tot_size = self.count_parameters()
        count = 0

        for param in self.parameters():
            sz = param.cpu().data.numpy().flatten().shape[0]
            raw = pvec[count:count + sz]
            reshaped = raw.reshape(param.cpu().data.numpy().shape)
            param.data = torch.from_numpy(reshaped).to(torch.device("cpu"))
            count += sz

        return pvec

    def count_parameters(self):
        count = 0
        for param in self.parameters():
            count += param.cpu().data.numpy().flatten().shape[0]
        return count

    def extract_grad(self):
        tot_size = self.count_parameters()
        pvec = np.zeros(tot_size, np.float32)
        count = 0
        for param in self.parameters():
            sz = param.grad.cpu().data.numpy().flatten().shape[0]
            pvec[count:count + sz] = param.grad.cpu().data.numpy().flatten()
            count += sz
        return pvec.copy()



    def initial_wieghts(self):

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                #m.weight.data.normal_(0, math.sqrt(2. / n))
                m.weight.data.fill_(np.random.random() )
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(np.random.random() )
                m.bias.data.zero_()
            elif isinstance(m, nn.Linear):
                m.bias.data.fill_(np.random.random() )



    #BRING MUTATION INTO THE NET CLASS



    #TODO implement safe mutations



