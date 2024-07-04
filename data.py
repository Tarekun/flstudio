import h5py
import torch
from torchvision import datasets, transforms
from torch.utils.data import ConcatDataset, Dataset, DataLoader, random_split
import numpy as np
from omegaconf import DictConfig

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


def split_dataset(dataset, val_ratio: float, test_ratio: float):
    """Splits the training dataset into training subset and validation subset"""
    tot_size = len(dataset)
    val_size = int(val_ratio * tot_size)
    test_size = int(test_ratio * tot_size)
    train_size = tot_size - val_size - test_size

    train_subset, val_subset, test_subset = random_split(
        dataset, [train_size, val_size, test_size]
    )
    return train_subset, val_subset, test_subset


def _get_datasets(
    num_writers: int,
    val_ratio: float,
    test_ratio: float,
    only_digits: bool = False,
):
    """Retrieves the FEMNIST dataset at the HDF5 file"""

    # TODO: include digits dataset file and choose how to properly handle these
    dataset_file = "write_digits.hdf5" if only_digits else "write_all.hdf5"
    full_dataset = h5py.File(dataset_file, "r")
    writers = sorted(full_dataset.keys())[:num_writers]
    train_sets = []
    val_sets = []
    test_sets = []

    for writer in writers:
        images = full_dataset[writer]["images"][:]
        labels = full_dataset[writer]["labels"][:]
        dataset = FemnistWriterDataset(images, labels, transform=transform)

        train_subset, val_subset, test_subset = split_dataset(
            dataset, val_ratio, test_ratio
        )
        train_sets.append(train_subset)
        val_sets.append(val_subset)
        test_sets.append(test_subset)

    return train_sets, val_sets, test_sets


def get_dataloaders(
    data_cfg: DictConfig,
):
    """Instatiates and returns the DataLoaders for the FEMNIST dataset partitioned by user"""

    train_sets, val_sets, test_sets = _get_datasets(
        data_cfg.num_writers,
        data_cfg.val_ratio,
        data_cfg.test_ratio,
        data_cfg.only_digits,
    )
    num_centralized = int(data_cfg.hybrid_ratio * data_cfg.num_writers)
    #
    if num_centralized > 0:
        train_sets = [ConcatDataset(train_sets[:num_centralized])] + train_sets[
            num_centralized:
        ]
        val_sets = [ConcatDataset(val_sets[:num_centralized])] + val_sets[
            num_centralized:
        ]

    train_loaders = [
        DataLoader(train_set, batch_size=data_cfg.batch_size, shuffle=True)
        for train_set in train_sets
    ]
    val_loaders = [
        DataLoader(val_set, batch_size=data_cfg.batch_size, shuffle=False)
        for val_set in val_sets
    ]

    # collapse the test datasets into one to test the global model
    combined_test_dataset = ConcatDataset(test_sets)
    test_loader = DataLoader(
        combined_test_dataset, batch_size=data_cfg.batch_size, shuffle=False
    )
    return train_loaders, val_loaders, test_loader
