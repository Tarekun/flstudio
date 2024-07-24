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


class HarDataset(Dataset):
    def __load_x__(filename) -> list[list[float]]:
        data = []
        with open(filename, "r") as file:
            for line in file:
                floats = [float(num) for num in line.split()]
                data.append(floats)
        return torch.tensor(data, dtype=torch.float32)

    def __load_y__(filename) -> list[torch.Tensor]:
        labels = []
        with open(filename, "r") as file:
            for line in file:
                label = int(line)
                one_hot = torch.zeros(6, dtype=torch.float32)
                one_hot[label - 1] = label
                labels.append(one_hot)
        return labels

    def __init__(self, train=True, transform=None):
        variant = "train" if train else "test"
        self.x = HarDataset.__load_x__(f"data/X_{variant}.txt")
        self.y = HarDataset.__load_y__(f"data/y_{variant}.txt")
        self.transform = transform

        if len(self.x) != len(self.y):
            raise Exception(
                "Problem with data files: x [{len(self.x)}] and y [{len(self.y)}] length should be the same"
            )

    def __getitem__(self, index):
        target_data = self.y[index]
        input_data = self.x[index]
        if self.transform:
            input_data = self.transform(input_data)

        return input_data, target_data

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


def _get_femnist_datasets(
    num_writers: int,
    val_ratio: float,
    test_ratio: float,
    only_digits: bool = False,
) -> tuple[list[Dataset], list[Dataset], list[Dataset]]:
    """Retrieves the FEMNIST dataset at the HDF5 file"""

    # TODO: include digits dataset file and choose how to properly handle these
    dataset_file = "write_digits.hdf5" if only_digits else "write_all.hdf5"
    full_dataset = h5py.File(f"data/{dataset_file}", "r")
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


def _get_har_datasets(
    num_clients: int,
) -> tuple[list[Dataset], list[Dataset], list[Dataset]]:
    full_trainset = HarDataset(train=True)
    full_testset = HarDataset(train=False)

    train_size, test_size = len(full_trainset), len(full_testset)
    train_sizes = [train_size // num_clients] * num_clients
    test_sizes = [test_size // num_clients] * num_clients
    train_sizes[0] += len(full_trainset) % num_clients
    test_sizes[0] += len(full_testset) % num_clients

    train_splits = random_split(full_trainset, train_sizes)
    test_splits = random_split(full_testset, test_sizes)
    # this dataset is already split into 70% train and 30% test, no validation used
    return train_splits, [], test_splits


def _get_datasets(
    data_cfg: DictConfig,
) -> tuple[list[Dataset], list[Dataset], list[Dataset]]:
    if data_cfg.dataset == "femnist":
        train_sets, val_sets, test_sets = _get_femnist_datasets(
            data_cfg.num_clients,
            data_cfg.val_ratio,
            data_cfg.test_ratio,
            data_cfg.only_digits,
        )
    elif data_cfg.dataset == "har":
        train_sets, val_sets, test_sets = _get_har_datasets(
            data_cfg.num_clients,
        )
    else:
        raise ValueError(f"Unsupported dataset: {data_cfg.dataset}")

    return train_sets, val_sets, test_sets


def get_dataloaders(
    data_cfg: DictConfig,
) -> tuple[list[DataLoader], DataLoader]:
    """Instatiates and returns the DataLoaders for the FEMNIST dataset partitioned by user"""

    train_sets, _, test_sets = _get_datasets(data_cfg)
    num_centralized = int(data_cfg.hybrid_ratio * data_cfg.num_clients)
    #
    if num_centralized > 0:
        train_sets = [ConcatDataset(train_sets[:num_centralized])] + train_sets[
            num_centralized:
        ]

    train_loaders = [
        DataLoader(train_set, batch_size=data_cfg.batch_size, shuffle=True)
        for train_set in train_sets
    ]

    # collapse the test datasets into one to test the global model
    combined_test_dataset = ConcatDataset(test_sets)
    test_loader = DataLoader(
        combined_test_dataset, batch_size=data_cfg.batch_size, shuffle=False
    )
    return train_loaders, test_loader
