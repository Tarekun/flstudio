import flwr as fl
from flwr.server.strategy import FedAvg
from flwr.simulation import start_simulation
from parameters import writers_to_include, num_rounds
from data import get_dataloaders
from client import get_client_generator


train_loaders, val_loaders = get_dataloaders(
    num_writers=writers_to_include, hybrid_ratio=0.2
)
client_fn = get_client_generator(train_loaders, val_loaders)

strategy = FedAvg(
    fraction_fit=1,
    fraction_evaluate=1,
    # TODO: maybe use this to control the number of clients??
    # min_available_clients=writers_to_include,
    # TODO: evaluate global model
    # evaluate_fn=get_evaluate_fn(cfg.num_classes, testloader),
)
history = start_simulation(
    client_fn=client_fn,
    num_clients=len(train_loaders),
    config=fl.server.ServerConfig(num_rounds=num_rounds),
    strategy=strategy,
    client_resources={"num_cpus": 2, "num_gpus": 0.0},
)
