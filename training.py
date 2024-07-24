import torch
import torch.nn as nn
import torch.optim as optim
from omegaconf import DictConfig
from model import get_proper_model
from collections import OrderedDict


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


def evaluate_model(model: nn.Module, test_loader) -> float:
    correct = 0
    total = 0
    total_loss = 0.0
    # Put the model in evaluation mode ??
    model.eval()
    model.to(device)

    with torch.no_grad():
        for images, labels in test_loader:
            images = images.to(device)
            labels = labels.to(device)
            outputs = model(images)

            total_loss += criterion(outputs, labels)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)

            labels_indices = torch.argmax(labels, dim=1)
            correct += (predicted == labels_indices).sum().item()

    accuracy = correct / total
    print(f"Evaluation accuracy on the test dataset: {100 * accuracy:.2f}%")
    return total_loss, accuracy


def get_evaluation_fn(num_classes: int, dataset: str, test_loader):
    def evaluation_fn(server_round, parameters, config):
        model = get_proper_model(num_classes, dataset)
        params_dict = zip(model.state_dict().keys(), parameters)
        state_dict = OrderedDict({k: torch.Tensor(v) for k, v in params_dict})
        model.load_state_dict(state_dict, strict=True)

        loss, accuracy = evaluate_model(model, test_loader)
        return loss, {"accuracy": accuracy, "loss": loss}

    return evaluation_fn
