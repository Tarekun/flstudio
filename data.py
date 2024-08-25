import h5py
import torch
from torchvision import datasets, transforms
from torch.utils.data import ConcatDataset, Dataset, DataLoader, Subset, random_split
import numpy as np
from omegaconf import DictConfig
import random

femnist_transform = transforms.Compose(
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
        self.transform = transform if not transform is None else femnist_transform

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
        super().__init__()
        variant = "train" if train else "test"
        self.x = HarDataset.__load_x__(f"data/X_{variant}.txt")
        self.y = HarDataset.__load_y__(f"data/y_{variant}.txt")
        self.transform = transform

        if len(self.x) != len(self.y):
            raise Exception(
                f"Problem with data files: x [{len(self.x)}] and y [{len(self.y)}] length should be the same"
            )

    def __getitem__(self, index):
        target_data = self.y[index]
        input_data = self.x[index]

        if self.transform:
            input_data = self.transform(input_data)

        return input_data, target_data

    def __len__(self):
        return len(self.y)


def _split_dataset(dataset, val_ratio: float, test_ratio: float):
    """Splits the training dataset into training subset and validation subset"""
    tot_size = len(dataset)
    val_size = int(val_ratio * tot_size)
    test_size = int(test_ratio * tot_size)
    train_size = tot_size - val_size - test_size

    train_subset, val_subset, test_subset = random_split(
        dataset, [train_size, val_size, test_size]
    )
    return train_subset, val_subset, test_subset


def _biased_split(dataset, num_subsets, num_classes, bias_factor=0.0):
    def sort_by_class(dataset, num_classes):
        sorted_dataset = [[] for _ in range(num_classes)]
        for idx, (_, label) in enumerate(dataset):
            # assuming one-hot encoded labels
            class_index = torch.argmax(label).item()
            sorted_dataset[class_index].append(idx)
        return sorted_dataset

    sorted_dataset = sort_by_class(dataset, num_classes)
    subset_size = len(dataset) // num_subsets
    # stores sample indices for each subset
    subsets = [[] for _ in range(num_subsets)]

    for i in range(num_subsets):
        percentage_position = i / num_subsets
        dominant_class = int(percentage_position * num_classes)
        class_weights = np.array(
            [
                # class probability in function of the distance from the dominant_class
                # computed as: e ^ (-b * |dc - label|)
                # where b is the given bias factor and dc is the dominant class
                np.exp(-bias_factor * abs(dominant_class - label))
                for label in range(num_classes)
            ]
        )
        # apply bias factor to skew the distribution more
        # class_weights = class_weights**bias_factor
        # normalize in [0,1] to get valid probabilities
        class_weights /= class_weights.sum()

        for _ in range(subset_size):
            # randomly choose a class based on the current distribution
            chosen_class = np.random.choice(range(num_classes), p=class_weights)
            if sorted_dataset[chosen_class]:
                sample_index = random.choice(sorted_dataset[chosen_class])
                subsets[i].append(sample_index)
                sorted_dataset[chosen_class].remove(sample_index)

    remaining_indices = [
        sample_idx for class_samples in sorted_dataset for sample_idx in class_samples
    ]
    for i, idx in enumerate(remaining_indices):
        subsets[i % num_subsets].append(idx)

    subset_datasets = [Subset(dataset, indices) for indices in subsets]
    return subset_datasets


def _get_femnist_datasets(
    num_writers: int,
    val_ratio: float,
    test_ratio: float,
    only_digits: bool = False,
) -> tuple[list[Dataset], list[Dataset], list[Dataset]]:
    """Retrieves the FEMNIST dataset at the HDF5 file. It returns a 3 element tuple containing:
    - the list of training sets partitioned per client
    - the list of validation sets partitioned per client
    - the list of test sets partitioned per client"""

    dataset_file = "write_digits.hdf5" if only_digits else "write_all.hdf5"
    full_dataset = h5py.File(f"data/{dataset_file}", "r")
    writers = sorted(full_dataset.keys())[:num_writers]
    train_sets = []
    val_sets = []
    test_sets = []

    for writer in writers:
        images = full_dataset[writer]["images"][:]
        labels = full_dataset[writer]["labels"][:]
        client_dataset = FemnistWriterDataset(
            images, labels, transform=femnist_transform
        )

        train_subset, val_subset, test_subset = _split_dataset(
            client_dataset, val_ratio, test_ratio
        )
        train_sets.append(train_subset)
        val_sets.append(val_subset)
        test_sets.append(test_subset)

    full_dataset.close()
    return train_sets, val_sets, test_sets


def _get_har_datasets(
    num_clients: int,
    num_classes: int,
    bias_factor: float,
) -> tuple[list[Dataset], Dataset]:
    """Retrieves the HAR dataset. It returns a 2 element tuple containing:
    - the list of training sets partitioned per client
    - the full test set unpartitioned"""

    full_trainset = HarDataset(train=True)
    full_testset = HarDataset(train=False)

    train_size = len(full_trainset)
    train_sizes = [train_size // num_clients] * num_clients
    train_sizes[0] += train_size % num_clients

    train_splits = _biased_split(
        full_trainset, num_clients, num_classes, bias_factor=bias_factor
    )
    # this dataset is already split into 70% train and 30% test, no validation used
    return train_splits, full_testset


def _get_datasets(
    data_cfg: DictConfig,
) -> tuple[list[Dataset], list[Dataset], Dataset]:
    """Returns the configured dataset already split in train, val, and test. Returns a 3 element tuple containing:
    - the list of training sets partitioned per client
    - the list of validation sets partitioned per client
    - one single test set to evaluate the global model"""

    if data_cfg.dataset == "femnist":
        train_sets, val_sets, test_sets = _get_femnist_datasets(
            data_cfg.num_clients,
            data_cfg.val_ratio,
            data_cfg.test_ratio,
            data_cfg.only_digits,
        )
        return train_sets, val_sets, ConcatDataset(test_sets)

    elif data_cfg.dataset == "har":
        train_sets, test_set = _get_har_datasets(
            data_cfg.num_clients, data_cfg.num_classes, data_cfg.get("bias_factor", 0.0)
        )
        return train_sets, [], test_set

    else:
        raise ValueError(f"Unsupported dataset: {data_cfg.dataset}")


def _centralize_trainsets(train_sets, num_clients, hybrid_ratio):
    num_centralized = int(hybrid_ratio * num_clients)
    if num_centralized > 0:
        train_sets = [ConcatDataset(train_sets[:num_centralized])] + train_sets[
            num_centralized:
        ]

    return train_sets


def get_horizontal_dataloaders(
    data_cfg: DictConfig,
) -> tuple[list[DataLoader], DataLoader]:
    """Instatiates and returns the train and test DataLoaders for the configured dataset partitioned by user"""

    train_sets, _, test_set = _get_datasets(data_cfg)
    train_sets = _centralize_trainsets(
        train_sets, data_cfg.hybrid_ratio, data_cfg.num_clients
    )

    train_loaders = [
        DataLoader(train_set, batch_size=data_cfg.batch_size, shuffle=True)
        for train_set in train_sets
    ]
    test_loader = DataLoader(test_set, batch_size=len(test_set), shuffle=False)

    return train_loaders, test_loader


def get_vertical_dataloaders(data_cfg: DictConfig) -> tuple[DataLoader, DataLoader]:
    if data_cfg.dataset != "har":
        raise ValueError(
            f"Only vertical supported dataset is HAR, requested was: {data_cfg.dataset}"
        )

    train_set = HarDataset(train=True)
    test_set = HarDataset(train=False)

    # process the whole training set in one batch as the output of local models
    # will be an embedding used as input for the server model, which will be the one
    # compute the gradient's for both local and global model
    return DataLoader(train_set, batch_size=len(train_set), shuffle=True), DataLoader(
        test_set, batch_size=len(test_set), shuffle=False
    )


def extract_features(full_features, num_clients, cid):
    """Gets only the features of this specific client"""
    total_features = 561  # TODO: find a better way of setting this
    features_per_client = total_features // num_clients
    start_idx = cid * features_per_client
    # if it is the last client we return everything from start_idx to the end
    end_idx = (
        start_idx + features_per_client if cid < num_clients - 1 else total_features
    )

    return full_features[:, start_idx:end_idx]
