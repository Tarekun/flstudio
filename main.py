import hydra
from omegaconf import DictConfig
import flwr as fl
from flwr.server.strategy import FedAvg
from flwr.simulation import start_simulation

from data import get_horizontal_dataloaders, get_vertical_dataloaders
from client import get_horizontal_client_generator, get_vertical_client_generator
from training import get_evaluation_fn
from visualization import plot_simulation
from strategy import *


def horizontal_simulation(cfg: DictConfig):
    sim_cfg, train_cfg, data_cfg = cfg.sim_cfg, cfg.train_cfg, cfg.data_cfg
    num_classes = data_cfg.num_classes

    train_loaders, test_loader = get_horizontal_dataloaders(data_cfg)
    client_fn = get_horizontal_client_generator(
        num_classes, data_cfg.dataset, train_cfg, train_loaders
    )
    evaluate_fn = get_evaluation_fn(num_classes, data_cfg.dataset, test_loader)

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
    plot_simulation(history)


def vertical_simulation(cfg: DictConfig):
    sim_cfg, train_cfg, data_cfg = cfg.sim_cfg, cfg.train_cfg, cfg.data_cfg
    num_classes = data_cfg.num_classes

    train_loader, test_loadedr = get_vertical_dataloaders(data_cfg)
    client_fn = get_vertical_client_generator(
        train_cfg, data_cfg.num_clients, train_loader
    )
    # TODO: add test validation
    # evaluate_fn = get_evaluation_fn(num_classes, data_cfg.dataset, test_loader)

    strategy = VerticalFedAvg(data_cfg.num_clients, num_classes, train_loader)
    history = start_simulation(
        client_fn=client_fn,
        num_clients=data_cfg.num_clients,
        config=fl.server.ServerConfig(num_rounds=sim_cfg.num_rounds),
        strategy=strategy,
        client_resources={"num_cpus": sim_cfg.num_cpus, "num_gpus": sim_cfg.num_gpus},
    )
    # TODO: implement plotting for vertical partitioning too
    # plot_simulation(history.metrics)


@hydra.main(config_path="conf", config_name="femnist")
def main(cfg: DictConfig):
    if cfg.partitioning == "horizontal":
        horizontal_simulation(cfg)
    elif cfg.partitioning == "vertical":
        vertical_simulation(cfg)

    else:
        raise ValueError(
            f"Unsupported partitioning selected: {cfg.partitioning}. Only 'horizontal' and 'vertical' are accepted"
        )


if __name__ == "__main__":
    main()
