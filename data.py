import h5py
import torch
from torchvision import datasets, transforms
from torch.utils.data import Dataset, DataLoader, random_split
import numpy as np
from parameters import batch_size

transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.5,), (0.5,))  # Normalize the data to [-1, 1]
])


class FemnistWriterDataset(Dataset):
    def __init__(self, images: np.ndarray, labels: np.ndarray, transform=None):
        if len(images) != len(labels):
            raise Exception(
                f"Different number of images ({len(images)}) and labels ({len(labels)})"
            )
        self.x, self.y = images, labels
        self.transform = transform


    def __getitem__(self, index):
        target_data = self.y[index]
        input_data = np.array(self.x[index])
        if self.transform:
            input_data = self.transform(input_data)

        return input_data, torch.tensor(target_data, dtype=torch.int64)


    def __len__(self):
        return len(self.y)


def split_dataset(dataset, val_ratio = 0.1):
    val_size = int(val_ratio * len(dataset))
    train_size = len(dataset) - val_size
    train_subset, val_subset = random_split(dataset, [train_size, val_size])
    return train_subset, val_subset


def get_dataloaders(num_writers: int = 100, test_ratio: float = 0.1, val_ratio: float = 0.1):
    full_dataset = h5py.File('write_all.hdf5', 'r')
    writers = sorted(full_dataset.keys())[:num_writers]
    train_loaders = []
    val_loaders = []
    
    for writer in writers:
        images = full_dataset[writer]['images'][:]
        labels = full_dataset[writer]['labels'][:]

        dataset = FemnistWriterDataset(images, labels, transform=transform)
        train_subset, val_subset = split_dataset(dataset, val_ratio)
        train_loaders.append(
            DataLoader(train_subset, batch_size=batch_size, shuffle=True)
        )
        val_loaders.append(
            DataLoader(val_subset, batch_size=batch_size, shuffle=False)
        )
        print(f"writer {writer} with {len(train_subset)} training samples and {len(val_subset)} validation samples")

    return train_loaders, val_loaders
