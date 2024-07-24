import hydra
from omegaconf import DictConfig
import flwr as fl
from flwr.server.strategy import FedAvg
from flwr.simulation import start_simulation

from data import get_dataloaders
from client import get_client_generator
from training import get_evaluation_fn
from visualization import plot_simulation


@hydra.main(config_path="conf", config_name="har")
def main(cfg: DictConfig):
    sim_cfg, train_cfg, data_cfg = cfg.sim_cfg, cfg.train_cfg, cfg.data_cfg
    num_classes = data_cfg.num_classes

    train_loaders, _, test_loader = get_dataloaders(data_cfg)
    client_fn = get_client_generator(train_loaders, num_classes, train_cfg)
    evaluate_fn = get_evaluation_fn(num_classes, test_loader)

    strategy = FedAvg(
        fraction_fit=1,
        fraction_evaluate=0,
        evaluate_fn=evaluate_fn,
    )
    history = start_simulation(
        client_fn=client_fn,
        num_clients=len(train_loaders),
        config=fl.server.ServerConfig(num_rounds=sim_cfg.num_rounds),
        strategy=strategy,
        client_resources={"num_cpus": sim_cfg.num_cpus, "num_gpus": sim_cfg.num_gpus},
    )
    plot_simulation(history)


if __name__ == "__main__":
    main()
