import torch
import torch.nn as nn
import torch.nn.functional as F
from data import extract_features

featmaps = [32, 64, 128]
kernels = [3, 3, 3]
first_linear_size = featmaps[2] * kernels[2] * kernels[2]
linears = [512, 256, 62]

latent_vector_length = 100


class CnnEmnist(nn.Module):
    def __init__(self, num_classes: int):
        super(CnnEmnist, self).__init__()
        # reused pooling layer
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2, padding=0)

        # 3 convolutional layers
        self.conv1 = nn.Conv2d(
            in_channels=1, out_channels=featmaps[0], kernel_size=kernels[0], padding=1
        )
        self.conv2 = nn.Conv2d(
            in_channels=featmaps[0],
            out_channels=featmaps[1],
            kernel_size=kernels[1],
            padding=1,
        )
        self.conv3 = nn.Conv2d(
            in_channels=featmaps[1],
            out_channels=featmaps[2],
            kernel_size=kernels[2],
            padding=1,
        )

        # 2 fully connected layers
        self.fc1 = nn.Linear(first_linear_size, linears[0])
        self.fc2 = nn.Linear(linears[0], linears[1])

        # output layer
        self.fc3 = nn.Linear(linears[1], num_classes)

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = self.pool(F.relu(self.conv3(x)))

        # flatten
        x = x.view(-1, first_linear_size)

        x = F.relu(self.fc1(x))
        x = self.fc2(x)

        # final layer uses softmax as this is a classification problem
        out = F.log_softmax(self.fc3(x), dim=1)
        return out


class HarModel(nn.Module):
    def __init__(self, num_classes: int):
        super(HarModel, self).__init__()
        self.fc1 = nn.Linear(561, 50)
        self.fc4 = nn.Linear(50, num_classes)

    def forward(self, x):
        x = F.relu(self.fc1(x))

        out = F.softmax(self.fc4(x), dim=1)
        return out


class ClientVerticalModel(nn.Module):
    def __init__(self, num_features: int):
        super(ClientVerticalModel, self).__init__()
        self.input = nn.Linear(num_features, 100)
        self.latent = nn.Linear(100, latent_vector_length)

    def forward(self, x):
        x = F.relu(self.input(x))
        latent = F.relu(self.latent(x))
        return latent


class ServerVerticalModel(nn.Module):
    def __init__(self, num_clients, num_classes: int):
        super(ServerVerticalModel, self).__init__()
        self.input = nn.Linear(num_clients * latent_vector_length, 100)
        self.output = nn.Linear(100, num_classes)

    def forward(self, x):
        x = F.relu(self.input(x))
        out = F.softmax(self.output(x))
        return out


class FullVerticalModel(nn.Module):
    def __init__(
        self,
        client_models: list[ClientVerticalModel],
        server_model: ServerVerticalModel,
        num_clients: int,
    ):
        super(FullVerticalModel, self).__init__()
        self.client_models = client_models
        self.server_model = server_model
        self.num_clients = num_clients

    def forward(self, x):
        embeddings = []
        for cid, model in enumerate(self.client_models):
            extracted_features = extract_features(x, self.num_clients, cid)
            embeddings.append(model(extracted_features))

        embeddings_aggregated = torch.cat(embeddings, dim=1)
        return self.server_model(embeddings_aggregated)


def get_proper_model(num_classes: int, dataset: str, is_vertical: bool = False):
    if dataset == "femnist":
        return CnnEmnist(num_classes)
    elif dataset == "har":
        return HarModel(num_classes)
    else:
        raise ValueError(
            f"dataset {dataset} not supported, 'femnist' and 'har' are the only possible values"
        )
