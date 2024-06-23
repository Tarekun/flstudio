import torch.nn as nn
import torch.nn.functional as F

featmaps = [32, 64, 128]
kernels = [3, 3, 3]
first_linear_size = featmaps[2] * kernels[2] * kernels[2]
linears = [512, 256, 62]


class CnnEmnist(nn.Module):
    def __init__(self):
        super(CnnEmnist, self).__init__()
        # reused pooling layer
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2, padding=0)
        
        # 3 convolutional layers
        self.conv1 = nn.Conv2d(in_channels=1, out_channels=featmaps[0], kernel_size=kernels[0], padding=1)
        self.conv2 = nn.Conv2d(in_channels=featmaps[0], out_channels=featmaps[1], kernel_size=kernels[1], padding=1)
        self.conv3 = nn.Conv2d(in_channels=featmaps[1], out_channels=featmaps[2], kernel_size=kernels[2], padding=1)
        
        # 2 fully connected layers
        self.fc1 = nn.Linear(first_linear_size, linears[0])
        self.fc2 = nn.Linear(linears[0], linears[1])
        
        # output layer, 62 classes in EMNIST
        self.fc3 = nn.Linear(linears[1], linears[2]) 

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = self.pool(F.relu(self.conv3(x)))
        
        # flatten
        x = x.view(-1, first_linear_size)

        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        
        # final layer uses softmax as this is a classification problem
        out = F.log_softmax(self.fc3(x), dim=1)
        return out
    