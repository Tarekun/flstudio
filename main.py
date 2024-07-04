import hydra
from omegaconf import DictConfig
import flwr as fl
from flwr.server.strategy import FedAvg
from flwr.simulation import start_simulation

from data import get_dataloaders
from client import get_client_generator


@hydra.main(config_path="conf", config_name="digits")
def main(cfg: DictConfig):
    sim_cfg, train_cfg, data_cfg = cfg.sim_cfg, cfg.train_cfg, cfg.data_cfg
    num_classes = 10 if data_cfg.only_digits else 62

    train_loaders, val_loaders = get_dataloaders(data_cfg)
    client_fn = get_client_generator(train_loaders, val_loaders, num_classes, train_cfg)

    strategy = FedAvg(
        fraction_fit=1,
        fraction_evaluate=0,
        # TODO: maybe use this to control the number of clients??
        # min_available_clients=writers_to_include,
        # TODO: evaluate global model
        # evaluate_fn=get_evaluate_fn(cfg.num_classes, testloader),
    )
    history = start_simulation(
        client_fn=client_fn,
        num_clients=len(train_loaders),
        config=fl.server.ServerConfig(num_rounds=sim_cfg.num_rounds),
        strategy=strategy,
        client_resources={"num_cpus": sim_cfg.num_cpus, "num_gpus": sim_cfg.num_gpus},
    )


if __name__ == "__main__":
    main()
