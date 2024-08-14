from collections import OrderedDict
import flwr as fl
import torch
from training import train, evaluate_model, device
from models import *
from omegaconf import DictConfig
from data import extract_features
import hydra


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
        model,
        train_loader,
        train_cfg,
    ):
        self.model = model
        self.cid = cid
        self.num_clients = num_clients
        self.total_features = total_features
        self.train_loader = train_loader
        self.train_cfg = train_cfg
        self.optimizer = hydra.utils.instantiate(
            train_cfg.optimizer, params=self.model.parameters()
        )

        self.embedding = None
        self.forward_pass()

    def forward_pass(self):
        """Performs a forward pass through the whole train_loader and saves the result in self.embedding"""
        # assuming the dataloader has a batch size of the entire dataset
        for batch in self.train_loader:
            features, _ = batch  # ignoring labels
            extracted_features = extract_features(
                features.to(device), self.num_clients, self.cid
            )
            self.embedding = self.model(extracted_features)

    def get_parameters(self, config):
        return [val.cpu().numpy() for _, val in self.model.state_dict().items()]

    def fit(self, parameters, config):
        """In a vertical setting the 'fit' method is used to compute the latent vector representation
        of the clinet's features that are then passed to the central server"""
        self.forward_pass()

        return [self.embedding.detach().numpy()], len(self.train_loader), {}

    def evaluate(self, gradients, config):
        """In a vertical setting the 'evaluate' method is used to the backprop pass with the gradient
        computed and provided by the central server"""
        self.model.zero_grad()
        self.embedding.backward(torch.from_numpy(gradients[self.cid]))
        self.optimizer.step()
        return 0.0, 1, {}


def get_horizontal_client_generator(
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
    train_cfg: DictConfig, num_clients: int, client_models, train_loader
):
    pass
    # def client_generator(cid: str):
    #     return VerticalClient(
    #         int(cid),
    #         561,  # TODO: find a better way of setting this
    #         num_clients,
    #         client_models[int(cid)],
    #         train_loader,
    #         train_cfg,
    #     ).to_client()

    # return client_generator
