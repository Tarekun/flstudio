import torch
import torch.nn as nn
import torch.optim as optim
from omegaconf import DictConfig


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
device = torch.device("cpu")
criterion = nn.CrossEntropyLoss()


def validate(model: nn.Module, val_loader):
    # Validation step
    model.eval()
    val_loss = 0.0
    correct = 0
    total = 0
    with torch.no_grad():
        for images, labels in val_loader:
            images = images.to(device)
            labels = labels.to(device)

            outputs = model(images)
            loss = criterion(outputs, labels)
            val_loss += loss.item()

            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

    val_loss /= len(val_loader)
    val_accuracy = 100 * correct / total
    print(f"Validation Loss: {val_loss:.4f}, Validation Accuracy: {val_accuracy:.2f}%")


def train(model: nn.Module, train_loader, train_cfg: DictConfig, val_loader=None):
    optimizer = optim.SGD(
        model.parameters(), lr=train_cfg.lr, momentum=train_cfg.momentum
    )
    model.to(device)

    for epoch in range(train_cfg.epochs):
        running_loss = 0.0
        for images, labels in train_loader:
            images = images.to(device)
            labels = labels.to(device)

            # Forward pass
            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)

            # Backward pass and optimize
            loss.backward()
            optimizer.step()

            running_loss += loss.item()

        print(
            f"Epoch [{epoch+1}/{train_cfg.epochs}], Loss: {running_loss/len(train_loader):.4f}"
        )
        if val_loader:
            validate(model, val_loader)


def evaluate(model: nn.Module, test_loader) -> float:
    correct = 0
    total = 0
    total_loss = 0.0
    model.to(device)

    with torch.no_grad():
        for images, labels in test_loader:
            images = images.to(device)
            labels = labels.to(device)
            outputs = model(images)

            total_loss += criterion(outputs, labels)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

    accuracy = correct / total
    print(f"Accuracy of the network on the 10000 test images: {100 * accuracy} %")
    return total_loss, accuracy
