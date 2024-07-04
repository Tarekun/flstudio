from collections import OrderedDict
import flwr as fl
import torch
from training import train, evaluate
from model import CnnEmnist
from omegaconf import DictConfig


class FlowerClient(fl.client.NumPyClient):
    def __init__(self, train_loader, val_loader, model, train_cfg) -> None:
        super().__init__()

        self.train_loader = train_loader
        self.val_loader = val_loader
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

    def evaluate(self, parameters, config={}):
        self.set_parameters(parameters)
        loss, accuracy = evaluate(self.model, self.val_loader)
        return float(loss), len(self.val_loader), {"accuracy": accuracy}


def get_client_generator(
    train_loaders, val_loaders, num_classes, train_cfg: DictConfig
):
    def client_generator(cid: str):
        return FlowerClient(
            train_loader=train_loaders[int(cid)],
            val_loader=val_loaders[int(cid)],
            model=CnnEmnist(num_classes),
            train_cfg=train_cfg,
        ).to_client()

    return client_generator
