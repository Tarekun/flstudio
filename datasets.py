from torchvision import datasets, transforms
from torch.utils.data import DataLoader
from parameters import batch_size

# Define transformations for the training and testing sets
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.5,), (0.5,))  # Normalize the data to [-1, 1]
])

# standard MNIST
mnist_train_dataset = datasets.MNIST(root='./data', train=True, download=True, transform=transform)
mnist_test_dataset = datasets.MNIST(root='./data', train=False, download=True, transform=transform)

mnist_train_loader = DataLoader(mnist_train_dataset, batch_size=batch_size, shuffle=True)
mnist_test_loader = DataLoader(mnist_test_dataset, batch_size=batch_size, shuffle=False)


# Extended MNIST 
emnist_train_dataset = datasets.EMNIST(root='./data', split='balanced', train=True, download=True, transform=transform)
emnist_test_dataset = datasets.EMNIST(root='./data', split='balanced', train=False, download=True, transform=transform)

emnist_train_loader = DataLoader(emnist_train_dataset, batch_size=batch_size, shuffle=True)
emnist_test_loader = DataLoader(emnist_test_dataset, batch_size=batch_size, shuffle=False)