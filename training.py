import torch
import torch.nn as nn
import torch.optim as optim
from parameters import *
from datasets import mnist_train_loader, mnist_test_loader


def train(model: nn.Module):
    optimizer = optim.SGD(model.parameters(), lr=learning_rate, momentum=momentum)

    for epoch in range(num_epochs):
        running_loss = 0.0
        for images, labels in mnist_train_loader:
            # Zero the parameter gradients
            optimizer.zero_grad()
            
            images = images.reshape(-1, 28*28).to(device)
            labels = labels.to(device)
            # Forward pass
            outputs = model(images)
            loss = criterion(outputs, labels)
            
            # Backward pass and optimize
            loss.backward()
            optimizer.step()
            
            running_loss += loss.item()
        
        print(f'Epoch [{epoch+1}/{num_epochs}], Loss: {running_loss/len(mnist_train_loader):.4f}')


def evaluate(model: nn.Module):
    correct = 0
    total = 0
    with torch.no_grad():
        for images, labels in mnist_test_loader:
            images = images.reshape(-1, 28*28).to(device)
            labels = labels.to(device)
            outputs = model(images)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

    print(f'Accuracy of the network on the 10000 test images: {100 * correct / total} %')
