import torch.nn as nn
from parameters import mnist_classes, emnist_classes


class SimpleNN(nn.Module):
    def __init__(self):
        super(SimpleNN, self).__init__()
        self.hidden = nn.Linear(28 * 28, 400)
        self.output = nn.Linear(400, emnist_classes)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        out = self.hidden(x)
        out = self.sigmoid(out)
        return self.output(out)
