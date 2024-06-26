import torch
import torch.nn as nn


device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


# hyper parameters
learning_rate = 0.01
momentum = 0.9
num_epochs = 10
batch_size = 64
criterion = nn.CrossEntropyLoss()
