from typing import Optional
import flwr as fl
import torch
import torch.nn as nn
import torch.optim as optim
from flwr.common import ndarrays_to_parameters, parameters_to_ndarrays
from training import *
from models import latent_vector_length
import hydra


def parameters_to_embeddings(results):
    embedding_results = [
        torch.from_numpy(parameters_to_ndarrays(fit_result.parameters)[0])
        for _, fit_result in results
    ]
    embeddings_aggregated = torch.cat(embedding_results, dim=1)
    server_embedding = embeddings_aggregated.detach().requires_grad_()

    return server_embedding


def gradients_to_parameters(server_embedding, num_clients):
    grads = server_embedding.grad.split([latent_vector_length] * num_clients, dim=1)
    grads = [grad.numpy() for grad in grads]
    return ndarrays_to_parameters(grads)


class VerticalFedAvg(fl.server.strategy.FedAvg):
    def __init__(
        self,
        num_clients: int,
        server_model: nn.Module,
        client_models: list[nn.Module],
        train_loader,
        train_cfg,
        *,
        fraction_fit: float = 1,
        fraction_evaluate: float = 1,
        min_fit_clients: int = 2,
        min_evaluate_clients: int = 2,
        min_available_clients: int = 2,
        evaluate_fn: Optional[callable] = None,
        on_fit_config_fn: Optional[callable] = None,
        on_evaluate_config_fn: Optional[callable] = None,
        accept_failures: bool = True,
        initial_parameters: Optional[any] = None,
        fit_metrics_aggregation_fn: Optional[callable] = None,
        evaluate_metrics_aggregation_fn: Optional[callable] = None,
    ) -> None:
        super().__init__(
            fraction_fit=fraction_fit,
            fraction_evaluate=fraction_evaluate,
            min_fit_clients=min_fit_clients,
            min_evaluate_clients=min_evaluate_clients,
            min_available_clients=min_available_clients,
            evaluate_fn=evaluate_fn,
            on_fit_config_fn=on_fit_config_fn,
            on_evaluate_config_fn=on_evaluate_config_fn,
            accept_failures=accept_failures,
            initial_parameters=initial_parameters,
            fit_metrics_aggregation_fn=fit_metrics_aggregation_fn,
            evaluate_metrics_aggregation_fn=evaluate_metrics_aggregation_fn,
        )
        self.server_model = server_model
        models = [server_model] + client_models
        self.optimizer = hydra.utils.instantiate(
            train_cfg.optimizer,
            params=[params for model in models for params in model.parameters()],
        )
        self.criterion = hydra.utils.instantiate(train_cfg.loss_fn)
        self.train_loader = train_loader
        self.num_clients = num_clients

    def aggregate_fit(
        self,
        rnd,
        results,
        failures,
    ):
        # do not aggregate if there are failures and failures are not accepted
        if not self.accept_failures and failures:
            return None, {}

        for batch in self.train_loader:
            _, labels = batch
            labels = labels.to(device)

            server_embedding = parameters_to_embeddings(results)

            output = self.server_model(server_embedding)
            loss = self.criterion(output, labels)
            loss.backward()
            self.optimizer.step()
            self.optimizer.zero_grad()

            parameters_aggregated = gradients_to_parameters(
                server_embedding, self.num_clients
            )

        with torch.no_grad():
            output = self.server_model(server_embedding)
            _, predicted = torch.max(output, dim=1)
            _, true_labels = torch.max(labels, dim=1)

            correct = (predicted == true_labels).sum().item()
            accuracy = correct / len(true_labels) * 100

        metrics = {"accuracy": accuracy, "loss": loss.item()}
        return parameters_aggregated, metrics

    def aggregate_evaluate(
        self,
        rnd,
        results,
        failures,
    ):
        metrics = {}
        return None, metrics
