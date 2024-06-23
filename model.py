import torch.nn as nn
import torch.nn.functional as F
from parameters import mnist_classes, emnist_classes


class CnnEmnist(nn.Module):
    def __init__(self):
        super(CnnEmnist, self).__init__()
        # reused pooling layer
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2, padding=0)
        
        # 3 convolutional layers
        self.conv1 = nn.Conv2d(in_channels=1, out_channels=32, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3, padding=1)
        self.conv3 = nn.Conv2d(in_channels=64, out_channels=128, kernel_size=3, padding=1)
        
        # 2 fully connected layers
        self.fc1 = nn.Linear(128 * 3 * 3, 512)
        self.fc2 = nn.Linear(512, 256)
        
        # output layer, 62 classes in EMNIST
        self.fc3 = nn.Linear(256, 62) 

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = self.pool(F.relu(self.conv3(x)))
        
        # flatten
        x = x.view(-1, 128 * 3 * 3)

        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        
        # final layer uses softmax as this is a classification problem
        out = F.log_softmax(self.fc3(x), dim=1)
        return out
    