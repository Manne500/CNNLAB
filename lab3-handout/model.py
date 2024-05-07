import torch
from torch import nn


class YourFirstNet(torch.nn.Module):
    def __init__(self, n_labels):
        super(YourFirstNet, self).__init__()

        # layer 1: Conv2 1->24, kernel 5, padding, same -> ReLU -> MaxPool2d
        conv1 = nn.Sequential(
            nn.Conv2d(3,24, kernel_size=5, padding="same"),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2)
        )
        
        # layer 2: Conv2 24->48, kernel 5, padding, same -> ReLU -> MaxPool2d
        conv2 = nn.Sequential(
            nn.Conv2d(24,48, kernel_size=5, padding="same"),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2)
        )
        
        # layer 3: Conv2 48->64, kernel 5, padding, same -> ReLU -> MaxPool2d
        conv3 = nn.Sequential(
            nn.Conv2d(48,64, kernel_size=5, padding="same"),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2,padding=1)
        )
    
        # classification layer 1: flatten
        flatten = nn.Flatten()
        
        # classification layer 2: linear from input to 256 -> ReLU
        fc1 = nn.Sequential(
            nn.Linear(64*23*23, 256),
            nn.ReLU(inplace=True)
        )
    
        # classification layer 3: linear from 256 to 10
        fc2 = nn.Linear(256, n_labels)

        self.cnn = nn.Sequential(
            conv1,
            conv2,
            conv3,
            flatten,
            fc1,
            fc2
        )
        
    def forward(self, X):
        return self.cnn(X)

class SimpleSeqCNN(torch.nn.Module):
    def __init__(self, n_labels,dropout):
        super(SimpleSeqCNN, self).__init__()
        print(f"Init model with Dropout: {dropout}")
        # layer 1: Conv2 1->24, kernel 5, padding, same -> ReLU -> MaxPool2d
        conv1 = nn.Sequential(
            nn.Conv2d(3,24, kernel_size=5, padding="same"),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
            nn.Dropout(dropout)
        )
        
        # layer 2: Conv2 24->48, kernel 5, padding, same -> ReLU -> MaxPool2d
        conv2 = nn.Sequential(
            nn.Conv2d(24,48, kernel_size=5, padding="same"),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
            nn.Dropout(dropout)
        )
        
        # layer 3: Conv2 48->64, kernel 5, padding, same -> ReLU -> MaxPool2d
        conv3 = nn.Sequential(
            nn.Conv2d(48,64, kernel_size=5, padding="same"),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2,padding=1),
            nn.Dropout(dropout)
        )
    
        # classification layer 1: flatten
        flatten = nn.Flatten()
        
        # classification layer 2: linear from input to 256 -> ReLU
        fc1 = nn.Sequential(
            nn.Linear(64*23*23, 256),
            nn.ReLU(inplace=True)
        )
    
        # classification layer 3: linear from 256 to 10
        fc2 = nn.Linear(256, n_labels)

        self.cnn = nn.Sequential(
            conv1,
            conv2,
            conv3,
            flatten,
            fc1,
            fc2
        )
        
    def forward(self, X):
        return self.cnn(X)

class ResNetJr(torch.nn.Module):
    def __init__(self, n_labels,dropout):
        super(ResNetJr, self).__init__()
        print(f"Init model with Dropout: {dropout}")
        # layer 1: Conv2 1->24, kernel 5, padding, same -> ReLU -> MaxPool2d
        self.conv1 = nn.Sequential(
            nn.Conv2d(3,24, kernel_size=5, padding="same"),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout)
        )
        self.pool = nn.MaxPool2d(2)
        self.pool_w_flatten = nn.MaxPool2d(2,padding=1)
        # layer 2: Conv2 24->48, kernel 5, padding, same -> ReLU -> MaxPool2d
        self.conv2 = nn.Sequential(
            nn.Conv2d(24,24, kernel_size=5, padding="same"),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Conv2d(24,24, kernel_size=5, padding="same"),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout)
        )
        self.conv_upscale=nn.Sequential(
            nn.Conv2d(24,48, kernel_size=5, padding="same"),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout)
        )
        
        # layer 3: Conv2 48->64, kernel 5, padding, same -> ReLU -> MaxPool2d
        self.conv3 = nn.Sequential(
            nn.Conv2d(48,48, kernel_size=3, padding="same"),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Conv2d(48,48, kernel_size=3, padding="same"),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout)
        )
    
        # classification layer 1: flatten
        flatten = nn.Flatten()
        
        # classification layer 2: linear from input to 256 -> ReLU
        fc1 = nn.Sequential(
            nn.Linear(48*23*23, 256),
            nn.ReLU(inplace=True)
        )
    
        # classification layer 3: linear from 256 to 10
        fc2 = nn.Linear(256, n_labels)
        self.endstuff=nn.Sequential(
            flatten,
            fc1,
            fc2

        )

        
    def forward(self, X):
        X=self.conv1(X)
        X=self.pool(X)
        X=X+self.conv2(X)
        X=self.pool(X)
        X=self.conv_upscale(X)
        X=X+self.conv3(X)
        X=self.pool_w_flatten(X)
        return self.endstuff(X)