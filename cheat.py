from data import get_horizontal_dataloaders, get_vertical_dataloaders
from client import get_horizontal_client_generator, get_vertical_client_generator
from training import get_horizontal_evaluation_fn, train, evaluate_model
from visualization import plot_simulations
from strategy import *
from models import *

from strategy import *
from models import *



@hydra.main(config_path="conf", config_name="femnist", version_base="1.2")
def main(cfg: DictConfig):
    train_loaders, test_loader = get_horizontal_dataloaders(cfg.data_cfg)
    model = CnnEmnist(62)
    accuracies = []
    losses = []

    print(len(train_loaders))
    for i in range(cfg.num_runs):
        print(f"Starting simulation number {i} of {cfg.num_runs}")
        for j in range(cfg.sim_cfg.num_rounds):
            print(f"Starting round {j} of {cfg.num_rounds}")

            train(model, train_loaders[0], cfg.train_cfg)
            loss, accuracy = evaluate_model(model, test_loader, cfg.train_cfg)
            accuracies.append(accuracy)
            losses.append(loss)

    plot_simulations((accuracies, losses))


if __name__ == "__main__":
    main()