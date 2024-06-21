import torch.nn as nn


class SimpleNN(nn.Module):
    def __init__(self):
        super(SimpleNN, self).__init__()
        self.hidden = nn.Linear(28 * 28, 400)
        self.output = nn.Linear(400, 10)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        out = self.hidden(x)
        out = self.sigmoid(out)
        return self.output(out)
