from flwr_datasets import FederatedDataset
from torch.utils.data import DataLoader
from torchvision.transforms import ToTensor
from flwr_datasets.utils import divide_dataset
from torchvision import datasets, transforms
from model import CnnEmnist
from parameters import device
from training import train, evaluate


# Load the centralized test dataset
# centralized_dataset = fds.load_split("test")


# Define transformations
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.5,), (0.5,))  # Normalize the data to [-1, 1]
])

def apply_transforms(batch):
    batch["img"] = [transform(img) for img in batch["image"]]
    return batch


def get_federated_dataset(dataset: str, clients: int) -> FederatedDataset:
    fds = FederatedDataset(dataset=dataset, partitioners={"train": clients})
    return fds


def get_transformed_partition(fds: FederatedDataset, partition_idx: int):
    partition = fds.load_partition(partition_idx, "train")
    return partition.with_transform(apply_transforms)


def get_partition_dataloaders(partition, batch_size: int) -> tuple[DataLoader]:
    train, valid, test = divide_dataset(partition, [0.8, 0.1, 0.1])
    train_loader = DataLoader(train, batch_size=batch_size, shuffle=True)
    valid_loader = DataLoader(valid, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test, batch_size=batch_size, shuffle=False)
    return train_loader, valid_loader, test_loader


fds = get_federated_dataset('mnist', 1)
partition = get_transformed_partition(fds, 0)
train_loader, val_loader, test_loader = get_partition_dataloaders(partition, 64)

conv_model = CnnEmnist().to(device)

train(conv_model, train_loader, val_loader)
evaluate(conv_model, test_loader)
