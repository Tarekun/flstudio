from model import SimpleNN, CnnEmnist
from training import train, evaluate
from parameters import device

# simple_model = SimpleNN().to(device)
conv_model = CnnEmnist().to(device)

train(conv_model)
evaluate(conv_model)
