import torch
from torchvision import datasets, transforms
from torch.utils.data import DataLoader, random_split, Dataset
from parameters import batch_size
import json
import os
import numpy as np
from collections import defaultdict
from typing import Any, DefaultDict, Dict, List, Tuple
from sklearn.model_selection import train_test_split
from preparation import *

# Define transformations for the training and testing sets
transform = transforms.Compose(
    [
        transforms.ToTensor(),
        transforms.Normalize((0.5,), (0.5,)),  # Normalize the data to [-1, 1]
    ]
)


def split_dataset(dataset, ratio=0.9):
    train_size = int(ratio * len(dataset))
    val_size = len(dataset) - train_size
    train_subset, val_subset = random_split(dataset, [train_size, val_size])
    return train_subset, val_subset


# standard MNIST
# mnist_train_dataset = datasets.MNIST(
#     root="./data", train=True, download=True, transform=transform
# )
# mnist_test_dataset = datasets.MNIST(
#     root="./data", train=False, download=True, transform=transform
# )
# mnist_train_subset, mnist_val_subset = split_dataset(mnist_train_dataset)


# mnist_train_loader = DataLoader(mnist_train_subset, batch_size=batch_size, shuffle=True)
# mnist_val_loader = DataLoader(mnist_val_subset, batch_size=batch_size, shuffle=True)
# mnist_test_loader = DataLoader(mnist_test_dataset, batch_size=batch_size, shuffle=False)


# Extended MNIST
# emnist_train_dataset = datasets.EMNIST(
#     root="./data", split="balanced", train=True, download=True, transform=transform
# )
# emnist_test_dataset = datasets.EMNIST(
#     root="./data", split="balanced", train=False, download=True, transform=transform
# )
# emnist_train_subset, emnist_val_subset = split_dataset(emnist_train_dataset)

# emnist_train_loader = DataLoader(
#     emnist_train_subset, batch_size=batch_size, shuffle=True
# )
# emnist_val_loader = DataLoader(emnist_val_subset, batch_size=batch_size, shuffle=True)
# emnist_test_loader = DataLoader(
#     emnist_test_dataset, batch_size=batch_size, shuffle=False
# )


def structure_dataset(data_path: str):
    def data_generator(files):
        for file_name in files:
            with open(os.path.join(data_path, file_name), "r") as file:
                dataset = json.load(file)
                user_data = dataset["user_data"]
                for user in user_data:
                    x_list, y_list = user_data[user]["x"], user_data[user]["y"]
                    for i in range(len(x_list)):
                        yield (x_list[i], y_list[i])

    files = [f for f in os.listdir(data_path) if f.endswith(".json")]
    files.sort()
    image_data = []

    data_gen = data_generator(files)

    for data in data_gen:
        image_data.append(data)

    print("fine")
    return image_data


class FemnistDataset(Dataset):
    """
    [LEAF: A Benchmark for Federated Settings](https://github.com/TalwalkarLab/leaf).

    We imported the preprocessing method for the Femnist dataset from GitHub.
    """

    def __init__(self, dataset, transform):
        self.x = dataset["x"]
        self.y = dataset["y"]
        self.transform = transform

    def __getitem__(self, index):
        """Retrieve the input data and its corresponding label at a given index.

        Args:
            index (int): The index of the data item to fetch.

        Returns
        -------
            tuple:
                - input_data (torch.Tensor): Reshaped and optionally transformed data.
                - target_data (int or torch.Tensor): Label for the input data.
        """
        input_data = np.array(self.x[index]).reshape(28, 28)
        if self.transform:
            input_data = self.transform(input_data)
        target_data = self.y[index]
        return input_data.to(torch.float32), target_data

    def __len__(self):
        """Return the number of labels present in the dataset.

        Returns
        -------
            int: The total number of labels.
        """
        return len(self.y)


def load_datasets(
    path: str,
) -> Tuple[DataLoader, DataLoader, DataLoader]:
    """Create the dataloaders to be fed into the model.

    Parameters
    ----------
    config: DictConfig
        data: float
            Used data type
        batch_size : int
            The size of the batches to be fed into the model,
            by default 10
        support_ratio : float
            The ratio of Support set for each client.(between 0 and 1)
            by default 0.2
    path : str
        The path where the leaf dataset was downloaded

    Returns
    -------
    Tuple[DataLoader, DataLoader, DataLoader]
    """

    data_type = "femnist"
    batch_size = 64

    print("load_datasets chiama partition_data")
    dataset = partition_data(data_type=data_type, dir_path=path, support_ratio=0.2)

    # Client list : 0.8, 0.1, 0.1
    clients_list = split_train_validation_test_clients(dataset[0]["users"])

    trainloaders: Dict[str, List[DataLoader]] = {"sup": [], "qry": []}
    valloaders: Dict[str, List[DataLoader]] = {"sup": [], "qry": []}
    testloaders: Dict[str, List[DataLoader]] = {"sup": [], "qry": []}

    if data_type == "femnist":
        transform = transforms.Compose([transforms.ToTensor()])
        for user in clients_list[0]:
            trainloaders["sup"].append(
                DataLoader(
                    FemnistDataset(dataset[0]["user_data"][user], transform),
                    batch_size=batch_size,
                    shuffle=True,
                )
            )
            trainloaders["qry"].append(
                DataLoader(
                    FemnistDataset(dataset[1]["user_data"][user], transform),
                    batch_size=batch_size,
                )
            )
        for user in clients_list[1]:
            valloaders["sup"].append(
                DataLoader(
                    FemnistDataset(dataset[0]["user_data"][user], transform),
                    batch_size=batch_size,
                )
            )
            valloaders["qry"].append(
                DataLoader(
                    FemnistDataset(dataset[1]["user_data"][user], transform),
                    batch_size=batch_size,
                )
            )
        for user in clients_list[2]:
            testloaders["sup"].append(
                DataLoader(
                    FemnistDataset(dataset[0]["user_data"][user], transform),
                    batch_size=batch_size,
                )
            )
            testloaders["qry"].append(
                DataLoader(
                    FemnistDataset(dataset[1]["user_data"][user], transform),
                    batch_size=batch_size,
                )
            )

    return trainloaders, valloaders, testloaders


load_datasets("../leaf/data/femnist/data")
