import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms
from torch.utils.data import DataLoader

# hyper parameters
learning_rate = 0.01
momentum = 0.9


class SimpleNN(nn.Module):
    def __init__(self):
        super(SimpleNN, self).__init__()
        self.hidden = nn.Linear(28 * 28, 400)
        self.output = nn.Linear(400, 10)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        flat = x.view(-1, 28 * 28)
        out = self.hidden(flat)
        out = self.sigmoid(out)
        return self.output(out)
    

simple_model = SimpleNN()

# Define the loss function and optimizer
criterion = nn.CrossEntropyLoss()
optimizer = optim.SGD(simple_model.parameters(), lr=learning_rate, momentum=momentum)

# Train the model
num_epochs = 10