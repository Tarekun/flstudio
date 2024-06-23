from torchvision import datasets, transforms
from torch.utils.data import DataLoader, random_split
from parameters import batch_size

# Define transformations for the training and testing sets
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.5,), (0.5,))  # Normalize the data to [-1, 1]
])


def split_dataset(dataset, ratio = 0.9):
    train_size = int(ratio * len(dataset))
    val_size = len(dataset) - train_size
    train_subset, val_subset = random_split(dataset, [train_size, val_size])
    return train_subset, val_subset


# standard MNIST
mnist_train_dataset = datasets.MNIST(root='./data', train=True, download=True, transform=transform)
mnist_test_dataset = datasets.MNIST(root='./data', train=False, download=True, transform=transform)
mnist_train_subset, mnist_val_subset = split_dataset(mnist_train_dataset)

mnist_train_loader = DataLoader(mnist_train_subset, batch_size=batch_size, shuffle=True)
mnist_val_loader = DataLoader(mnist_val_subset, batch_size=batch_size, shuffle=True)
mnist_test_loader = DataLoader(mnist_test_dataset, batch_size=batch_size, shuffle=False)


# Extended MNIST 
emnist_train_dataset = datasets.EMNIST(root='./data', split='balanced', train=True, download=True, transform=transform)
emnist_test_dataset = datasets.EMNIST(root='./data', split='balanced', train=False, download=True, transform=transform)
emnist_train_subset, emnist_val_subset = split_dataset(emnist_train_dataset)

emnist_train_loader = DataLoader(emnist_train_subset, batch_size=batch_size, shuffle=True)
emnist_val_loader = DataLoader(emnist_val_subset, batch_size=batch_size, shuffle=True)
emnist_test_loader = DataLoader(emnist_test_dataset, batch_size=batch_size, shuffle=False)