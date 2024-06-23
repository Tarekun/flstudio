from torchvision import datasets, transforms
from torch.utils.data import DataLoader, random_split
from parameters import batch_size

# Define transformations for the training and testing sets
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.5,), (0.5,))  # Normalize the data to [-1, 1]
])

# standard MNIST
mnist_train_dataset = datasets.MNIST(root='./data', train=True, download=True, transform=transform)
mnist_test_dataset = datasets.MNIST(root='./data', train=False, download=True, transform=transform)
train_size = int(0.9 * len(mnist_train_dataset))
val_size = len(mnist_train_dataset) - train_size
mnist_train_subset, mnist_val_subset = random_split(mnist_train_dataset, [train_size, val_size])

mnist_train_loader = DataLoader(mnist_train_subset, batch_size=batch_size, shuffle=True)
mnist_val_loader = DataLoader(mnist_val_subset, batch_size=batch_size, shuffle=True)
mnist_test_loader = DataLoader(mnist_test_dataset, batch_size=batch_size, shuffle=False)


# Extended MNIST 
emnist_train_dataset = datasets.EMNIST(root='./data', split='balanced', train=True, download=True, transform=transform)
emnist_test_dataset = datasets.EMNIST(root='./data', split='balanced', train=False, download=True, transform=transform)
train_size = int(0.9 * len(emnist_train_dataset))
val_size = len(emnist_train_dataset) - train_size
emnist_train_subset, emnist_val_subset = random_split(emnist_train_dataset, [train_size, val_size])


emnist_train_loader = DataLoader(emnist_train_subset, batch_size=batch_size, shuffle=True)
emnist_val_loader = DataLoader(emnist_val_subset, batch_size=batch_size, shuffle=True)
emnist_test_loader = DataLoader(emnist_test_dataset, batch_size=batch_size, shuffle=False)