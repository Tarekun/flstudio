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
        super().__init__()
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
    def __init__(self, features, labels, train=True, transform=None):
        super().__init__()
        if len(features) != len(labels):
            raise Exception(
                f"Different number of features ({len(features)}) and labels ({len(labels)})"
            )
        self.x, self.y = features, labels
        self.transform = transform

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


def _load_har_dataset(train: bool):
    variant = "train" if train else "test"
    full_features = [[] for _ in range(30)]
    full_labels = [[] for _ in range(30)]

    # get subject ids for measurements
    with open(f"data/subject_{variant}.txt", "r") as subject_file:
        subject_ids = [int(line.strip()) - 1 for line in subject_file]
    # get features and match them with the user
    with open(f"data/X_{variant}.txt", "r") as feature_file:
        for i, line in enumerate(feature_file):
            features = [float(num) for num in line.split()]
            full_features[subject_ids[i]].append(features)
    # get labels and match them with the user
    with open(f"data/y_{variant}.txt", "r") as label_file:
        for i, line in enumerate(label_file):
            label = int(line)
            one_hot = torch.zeros(6, dtype=torch.float32)
            one_hot[label - 1] = label
            full_labels[subject_ids[i]].append(one_hot)

    if train:
        return full_features, full_labels
    # flatten in one dataset for testing
    else:
        return [
            feature for user_features in full_features for feature in user_features
        ], [label for user_labels in full_labels for label in user_labels]


def _get_har_datasets() -> tuple[list[Dataset], Dataset]:
    """Retrieves the HAR dataset. It returns a 2 element tuple containing:
    - the list of training sets partitioned per client
    - the full test set unpartitioned"""

    train_features, train_labels = _load_har_dataset(True)
    test_features, test_labels = _load_har_dataset(False)

    client_datasets = [
        HarDataset(
            features=torch.tensor(train_features[cid], dtype=torch.float32),
            labels=train_labels[cid],
        )
        for cid in range(len(train_features))
        # some clients dont include measurements in the trainset
        if len(train_features[cid]) > 0
    ]
    test_dataset = HarDataset(
        torch.tensor(test_features, dtype=torch.float32), test_labels
    )
    return client_datasets, test_dataset


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
        train_sets, test_set = _get_har_datasets()
        return train_sets, [], test_set

    else:
        raise ValueError(f"Unsupported dataset: {data_cfg.dataset}")


def _centralize_trainsets(
    train_sets: list[Dataset], num_clients: int, hybrid_ratio, hybrid_method
):
    if hybrid_method == "unify":
        # this method collapses the first num_centralized datasets into one
        # TODO: handle permutation somewhere else?
        num_centralized = int(hybrid_ratio * num_clients)
        shared_sets = []
        for _ in range(num_centralized):
            random_index = random.randint(0, len(train_sets) - 1)
            shared_sets.append(train_sets.pop(random_index))

        if num_centralized > 0:
            train_sets = [ConcatDataset(shared_sets)] + train_sets

    elif hybrid_method == "share":
        # this method copies num_shared samples in a collective dataset
        shared_sets = []
        for train_set in train_sets:
            set_size = len(train_set)
            num_shared = int(hybrid_ratio * set_size)
            if num_shared > 0:
                # ignore the unshared part as we do not change local datasets here
                shared_set, _ = random_split(
                    train_set, [num_shared, set_size - num_shared]
                )
                shared_sets.append(shared_set)
        if len(shared_sets) > 0:
            train_sets = train_sets + [ConcatDataset(shared_sets)]

    elif hybrid_method == "share-disjoint":
        # this method removes num_shared samples from clients and puts them in the collective
        shared_sets = []
        client_sets = []
        for train_set in train_sets:
            set_size = len(train_set)
            num_shared = int(hybrid_ratio * set_size)
            if num_shared > 0:
                shared_set, unshared_set = random_split(
                    train_set, [num_shared, set_size - num_shared]
                )
                shared_sets.append(shared_set)
                client_sets.append(unshared_set)
            else:
                # dont lose clients sets!
                client_sets.append(train_set)
        if len(shared_sets) > 0:
            train_sets = client_sets + [ConcatDataset(shared_sets)]

    return train_sets


def get_horizontal_dataloaders(
    data_cfg: DictConfig,
) -> tuple[list[DataLoader], DataLoader]:
    """Instatiates and returns the train and test DataLoaders for the configured dataset partitioned by user"""

    train_sets, _, test_set = _get_datasets(data_cfg)
    train_sets = _centralize_trainsets(
        train_sets=train_sets,
        num_clients=len(train_sets),
        hybrid_ratio=data_cfg.hybrid_ratio,
        hybrid_method=data_cfg.hybrid_method,
    )
    print("Client datasets produced:")
    for train_set in train_sets:
        print(len(train_set), end=" ")

    train_loaders = [
        DataLoader(train_set, batch_size=data_cfg.batch_size, shuffle=True)
        for train_set in train_sets
    ]
    test_loader = DataLoader(test_set, batch_size=len(test_set), shuffle=False)

    return train_loaders, test_loader


def get_vertical_dataloaders(data_cfg: DictConfig) -> tuple[DataLoader, DataLoader]:
    # TODO: this need to be fixed now
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
