import torch
import torch.nn as nn
import torch.nn.functional as F

class LumberjackNet(nn.Module):
    def __init__(self,p=0.5,chan1=32,kernel=5, hop_length=512, n_mfcc=40, n_fft=512):
        super(LumberjackNet, self).__init__()
        self.hop_length = hop_length
        self.n_mfcc = n_mfcc
        self.n_fft = n_fft
        self.signal_length = 72050
        self.inputH = 200
        self.inputW = 643
        self.p = p
        self.kernel = kernel
        # channels, output, kernels 
        self.conv1 = nn.Conv2d(1,chan1,kernel)
        # kernel size, stride
        self.pool = nn.MaxPool2d(2,2) # chan1, n_mfcc-k+1, 141-k+1
        # # channels, output, kernels # 32,18,67  
        self.conv2W     = int((self.inputW-self.kernel+1) / 2)
        self.conv2H     = int((self.inputH-self.kernel+1) / 2)
        self.conv2      = nn.Conv2d(chan1,self.conv2W,kernel) # channels, 40 - kernel%2 / pool, 141 - kernel%2 / pool
        self.fc1mul2    = int((self.conv2W-self.kernel+1)/2)
        self.fc1uml3    = int((self.conv2H-self.kernel+1)/2)
        self.fc1        = nn.Linear(self.conv2W*self.fc1mul2*self.fc1uml3,100)
        self.fc2        = nn.Linear(100,10)
        self.fc3        = nn.Linear(10,1)
        self.drop       = nn.Dropout(p=p)

    def forward(self,x):
        x = x.view(-1,1,10272,self.inputH)                # -> n, 1, 40, 141
        x = self.pool(F.relu(self.conv1(x))) # -> n, 6, 18, 68
        x = self.pool(F.relu(self.conv2(x))) # -> n, 18, 7, 32
        x = x.view(-1,self.conv2W*self.fc1mul2*self.fc1uml3)               # -> n, 18*7*32
        x = F.relu(self.fc1(x))              # -> n, 100
        x = F.relu(self.fc2(x))              # -> n, 84
        x = self.fc3(x)                      # -> n, 1
        return x

