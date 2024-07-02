import torch
import torch.nn as nn


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
device = torch.device("cpu")
writers_to_include = 100
num_rounds = 5


# hyper parameters
learning_rate = 0.01
momentum = 0.9
num_epochs = 6
batch_size = 32
criterion = nn.CrossEntropyLoss()
