import hydra
from omegaconf import DictConfig, OmegaConf
import flwr as fl
from flwr.server.strategy import FedAvg
from flwr.simulation import start_simulation

from data import get_horizontal_dataloaders, get_vertical_dataloaders
from client import get_horizontal_client_generator, get_vertical_client_generator
from training import get_horizontal_evaluation_fn
from visualization import plot_simulations
from strategy import *
from models import *

from strategy import *
from models import *


def horizontal_simulation(cfg: DictConfig):
    sim_cfg, train_cfg, data_cfg = cfg.sim_cfg, cfg.train_cfg, cfg.data_cfg
    num_classes = data_cfg.num_classes

    train_loaders, test_loader = get_horizontal_dataloaders(data_cfg)
    client_fn = get_horizontal_client_generator(
        num_classes, data_cfg.dataset, train_cfg, train_loaders
    )
    evaluate_fn = get_horizontal_evaluation_fn(
        num_classes, data_cfg.dataset, test_loader, train_cfg
    )

    strategy = FedAvg(
        fraction_fit=1,
        fraction_evaluate=0,
        evaluate_fn=evaluate_fn,
    )

    history = start_simulation(
        client_fn=client_fn,
        # should use this instead of data_cfg.num_clients because in case centralization
        # of the dataset is being used len(train_loaders) will be strictly less than it
        num_clients=len(train_loaders),
        config=fl.server.ServerConfig(num_rounds=sim_cfg.num_rounds),
        strategy=strategy,
        client_resources={"num_cpus": sim_cfg.num_cpus, "num_gpus": sim_cfg.num_gpus},
    )
    return history


def vertical_simulation(cfg: DictConfig):
    sim_cfg, train_cfg, data_cfg = cfg.sim_cfg, cfg.train_cfg, cfg.data_cfg
    num_classes = data_cfg.num_classes

    num_features = 561  # TODO: find a better way of setting this
    features_per_client = num_features // data_cfg.num_clients
    remainder = num_features % data_cfg.num_clients

    # all models are defined here so that both clients and server can use the same models
    client_models = [
        ClientVerticalModel(features_per_client)
        for _ in range(data_cfg.num_clients - 1)
    ]
    client_models.append(ClientVerticalModel(features_per_client + remainder))
    server_model = ServerVerticalModel(data_cfg.num_clients, num_classes)

    train_loader, test_loader = get_vertical_dataloaders(data_cfg)
    client_fn = get_vertical_client_generator(
        train_cfg, data_cfg.num_clients, client_models, train_loader
    )
    evaluate_fn = get_vertical_evaluation_fn(
        client_models, server_model, data_cfg.num_clients, test_loader, train_cfg
    )

    strategy = VerticalFedAvg(
        data_cfg.num_clients,
        server_model,
        client_models,
        train_loader,
        train_cfg,
        evaluate_fn=evaluate_fn,
    )

    history = start_simulation(
        client_fn=client_fn,
        num_clients=data_cfg.num_clients,
        config=fl.server.ServerConfig(num_rounds=sim_cfg.num_rounds),
        strategy=strategy,
        client_resources={"num_cpus": sim_cfg.num_cpus, "num_gpus": sim_cfg.num_gpus},
    )
    return history


@hydra.main(config_path="conf", config_name="femnist", version_base="1.2")
def main(cfg: DictConfig):
    histories = []
    partitioning = cfg.partitioning

    for i in range(cfg.num_runs):
        print(
            f"Starting simulation number {i} of {cfg.num_runs} with the following config:"
        )
        print(OmegaConf.to_yaml(cfg))

        if partitioning == "horizontal":
            histories.append(horizontal_simulation(cfg))
        elif partitioning == "vertical":
            histories.append(vertical_simulation(cfg))

        else:
            raise ValueError(
                f"Unsupported partitioning selected: {partitioning}. Only 'horizontal' and 'vertical' are accepted"
            )

    print("Plotting results...")
    plot_simulations(
        histories,
        cfg,
        dir_name=f"{cfg.data_cfg.dataset}-{partitioning}",
    )


if __name__ == "__main__":
    main()
