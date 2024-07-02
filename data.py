import h5py
import torch
from torchvision import datasets, transforms
from torch.utils.data import ConcatDataset, Dataset, DataLoader, random_split
import numpy as np
from parameters import batch_size

transform = transforms.Compose(
    [
        transforms.ToTensor(),
        transforms.Normalize((0.5,), (0.5,)),  # Normalize the data to [-1, 1]
    ]
)


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


def split_dataset(dataset, val_ratio=0.1):
    val_size = int(val_ratio * len(dataset))
    train_size = len(dataset) - val_size
    train_subset, val_subset = random_split(dataset, [train_size, val_size])
    return train_subset, val_subset


def _get_datasets(
    num_writers: int = 100,
    val_ratio: float = 0.1,
    only_digits: bool = False,
):
    # TODO: include digits dataset file and choose how to properly handle these
    dataset_file = "" if only_digits else "write_all.hdf5"
    full_dataset = h5py.File(dataset_file, "r")
    writers = sorted(full_dataset.keys())[:num_writers]
    train_sets = []
    val_sets = []

    for writer in writers:
        images = full_dataset[writer]["images"][:]
        labels = full_dataset[writer]["labels"][:]
        dataset = FemnistWriterDataset(images, labels, transform=transform)

        train_subset, val_subset = split_dataset(dataset, val_ratio)
        train_sets.append(train_subset)
        val_sets.append(val_subset)

    return train_sets, val_sets


def get_dataloaders(
    num_writers: int = 100,
    # TODO: to be supported
    test_ratio: float = 0.1,
    val_ratio: float = 0.1,
    only_digits: bool = False,
    hybrid_ratio: float = 0.0,
):
    train_sets, val_sets = _get_datasets(num_writers, val_ratio, only_digits)
    num_centralized = int(hybrid_ratio * num_writers)
    #
    if num_centralized > 0:
        train_sets = [ConcatDataset(train_sets[:num_centralized])] + train_sets[
            num_centralized:
        ]
        val_sets = [ConcatDataset(val_sets[:num_centralized])] + val_sets[
            num_centralized:
        ]

    train_loaders = [
        DataLoader(train_set, batch_size=batch_size, shuffle=True)
        for train_set in train_sets
    ]
    val_loaders = [
        DataLoader(val_set, batch_size=batch_size, shuffle=False)
        for val_set in val_sets
    ]
    return train_loaders, val_loaders
