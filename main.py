from model import SimpleNN
from training import train, evaluate
from parameters import device

simple_model = SimpleNN().to(device)

train(simple_model)
evaluate(simple_model)
