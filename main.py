from model import CnnEmnist
from training import train, evaluate
from parameters import device

conv_model = CnnEmnist().to(device)

train(conv_model)
evaluate(conv_model)
