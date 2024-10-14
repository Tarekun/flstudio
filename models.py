import torch
import torch.nn as nn
import torch.nn.functional as F
from data import extract_features


latent_vector_length = 20


class CnnEmnist(nn.Module):
    def __init__(self, num_classes: int):
        super(CnnEmnist, self).__init__()
        # reused pooling layer
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2, padding=0)

        # 2 convolutional layers
        self.conv1 = nn.Conv2d(in_channels=1, out_channels=32, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(
            in_channels=32,
            out_channels=64,
            kernel_size=3,
            padding=1,
        )

        # fully connected layer
        self.fc = nn.Linear(64 * 7 * 7, 512)

        # output layer
        self.classification = nn.Linear(512, num_classes)

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))

        # flatten
        x = x.view(-1, 64 * 7 * 7)

        x = F.relu(self.fc(x))

        # final layer uses softmax as this is a classification problem
        out = F.log_softmax(self.classification(x), dim=1)
        return out


class HarModel(nn.Module):
    def __init__(self, num_classes: int):
        super(HarModel, self).__init__()
        self.fc1 = nn.Linear(561, 50)
        self.fc4 = nn.Linear(50, num_classes)

    def forward(self, x):
        x = F.relu(self.fc1(x))

        out = F.log_softmax(self.fc4(x), dim=1)
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
        out = F.log_softmax(self.output(x), dim=1)
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


def get_proper_model(num_classes: int, dataset: str):
    if dataset == "femnist":
        return CnnEmnist(num_classes)
    elif dataset == "har":
        return HarModel(num_classes)
    else:
        raise ValueError(
            f"dataset {dataset} not supported, 'femnist' and 'har' are the only possible values"
        )
