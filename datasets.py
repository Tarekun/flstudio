from torchvision import datasets, transforms
from torch.utils.data import DataLoader

# Define transformations for the training and testing sets
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.5,), (0.5,))  # Normalize the data to [-1, 1]
])

# Download and load the training and test datasets
mnist_train_dataset = datasets.MNIST(root='./data', train=True, download=True, transform=transform)
mnist_test_dataset = datasets.MNIST(root='./data', train=False, download=True, transform=transform)

mnist_train_loader = DataLoader(mnist_train_dataset, batch_size=64, shuffle=True)
mnist_test_loader = DataLoader(mnist_test_dataset, batch_size=64, shuffle=False)
