from model import CnnEmnist
from training import train, evaluate
from parameters import device
from data import emnist_train_loader, emnist_val_loader, emnist_test_loader

conv_model = CnnEmnist().to(device)

train(conv_model, emnist_train_loader, emnist_val_loader)
evaluate(conv_model, emnist_test_loader)
