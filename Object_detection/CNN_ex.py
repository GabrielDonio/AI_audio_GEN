import torch
from torch import nn, optim
import torch.nn.functional as F
from torch.utils.data import DataLoader
import torchvision.transforms as transforms
import torchvision.datasets as datasets

class SimpleCNN(nn.Module):
    def __init__(self,in_channels,num_classes):
        super(SimpleCNN,self).__init__()
        self.conv1 = nn.Conv2d(in_channels,out_channels=8,kernel_size=(3,3),stride=(1,1),padding=(1,1))
        #nb_output_features = int((nb_input_features + 2*padding - kernel_size) / stride) + 1
        self.pool = nn.MaxPool2d(kernel_size=(2,2),stride=(2,2))
        self.conv2 = nn.Conv2d(in_channels=8,out_channels=16,kernel_size=(3,3),stride=(1,1),padding=(1,1))
        self.fc1 = nn.Linear(16*7*7, num_classes)
    def forward(self,x):
        x=self.conv1(x)
        x=F.relu(x)
        x=self.pool(x)
        x=self.conv2(x)
        x=F.relu(x)
        x=self.pool(x)
        x=x.reshape(x.shape[0], -1) 
        x=self.fc1(x)
        return x

