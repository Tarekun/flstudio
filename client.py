from collections import OrderedDict
import flwr as fl
import torch
from training import train, evaluate_model, device
from models import *
from omegaconf import DictConfig


class HorizontalClient(fl.client.NumPyClient):
    def __init__(self, train_loader, model, train_cfg) -> None:
        super().__init__()
        self.train_loader = train_loader
        self.model = model
        self.train_cfg = train_cfg

    def set_parameters(self, parameters):
        params_dict = zip(self.model.state_dict().keys(), parameters)
        state_dict = OrderedDict({k: torch.Tensor(v) for k, v in params_dict})
        self.model.load_state_dict(state_dict, strict=True)

    def get_parameters(self, config={}):
        return [val.cpu().numpy() for _, val in self.model.state_dict().items()]

    def fit(self, parameters, config={}):
        self.set_parameters(parameters)
        train(self.model, self.train_loader, self.train_cfg)
        return self.get_parameters(), len(self.train_loader), {}


class VerticalClient(fl.client.NumPyClient):
    def __init__(
        self,
        cid: int,
        total_features: int,
        num_clients: int,
        train_loader,
        model,
        train_cfg,
    ):
        self.cid = cid
        self.num_clients = num_clients
        self.total_features = total_features
        self.model = model
        self.train_loader = train_loader
        self.train_cfg = train_cfg
        self.optimizer = torch.optim.SGD(self.model.parameters(), lr=0.01)
        self.embedding = None
        self.forward_pass()

    def forward_pass(self):
        # Assuming the dataloader has a batch size of the entire dataset
        for batch in self.train_loader:
            # ignoring labels
            features, _ = batch
            features = features.to(device)
            # extracting client's features
            features_per_client = self.total_features // self.num_clients
            start_idx = self.cid * features_per_client
            end_idx = start_idx + features_per_client
            extracted_features = features[:, start_idx:end_idx]
            self.embedding = self.model(extracted_features)

    def get_parameters(self, config):
        return [val.cpu().numpy() for _, val in self.model.state_dict().items()]

    def fit(self, parameters, config):
        """In a vertical setting the 'fit' method is used to compute the latent vector representation
        of the clinet's features that are then passed to the central server"""
        self.forward_pass()

        return [self.embedding.detach().numpy()], 1, {}

    def evaluate(self, gradients, config):
        """In a vertical setting the 'evaluate' method is used to the backprop pass with the gradient
        computed and provided by the central server"""
        self.model.zero_grad()
        self.embedding.backward(torch.from_numpy(gradients[self.cid]))
        self.optimizer.step()
        return 0.0, 1, {}


def get_client_generator(
    num_classes: int, dataset: str, train_cfg: DictConfig, train_loaders
):
    def client_generator(cid: str):
        return HorizontalClient(
            train_loader=train_loaders[int(cid)],
            model=get_proper_model(num_classes, dataset),
            train_cfg=train_cfg,
        ).to_client()

    return client_generator


def get_vertical_client_generator(
    train_cfg: DictConfig, num_clients: int, train_loader
):
    num_features = 561 // num_clients

    def client_generator(cid: str):
        return VerticalClient(
            int(cid),
            561,
            num_clients,
            train_loader,
            ClientVerticalModel(num_features),
            train_cfg,
        ).to_client()

    return client_generator
