import os
import matplotlib.pyplot as plt
from omegaconf import DictConfig, OmegaConf


PLOTS_DIR = "./plots"
os.makedirs(PLOTS_DIR, exist_ok=True)


def extract_metric_data(metric_history):
    return [data[1] for data in metric_history]


def _format_filename(cfg: DictConfig):
    name = "results"
    bias_factor = cfg.data_cfg.get("bias_factor", 0.0)
    name += f"-c{cfg.data_cfg.num_clients}"
    name += f"-h{cfg.data_cfg.hybrid_ratio}"
    name += f"-hm_{cfg.data_cfg.hybrid_method}"
    name += f"-b{bias_factor}"
    name += f"-lr{cfg.train_cfg.optimizer.lr}"
    name += f"-e{cfg.train_cfg.epochs}"
    name += f"-{cfg.train_cfg.optimizer._target_}"

    name = name.replace(".", "_")
    return f"{name}.png"


def create_plot(filename: str, x, y, title="", ylabel=""):
    # plt.figure(figsize=(8, 6))
    plt.plot(x, y, marker="o")
    plt.title(title)
    # plt.xlabel("Round")
    plt.ylabel(ylabel)
    plt.grid(True)
    plt.savefig(os.path.join(PLOTS_DIR, filename))
    plt.close()


def create_single_plot(filename: str, rounds, losses, accuracies):
    fig, ax1 = plt.subplots(figsize=(10, 6))

    # Plotting losses
    ax1.set_xlabel("Round")
    ax1.set_ylabel("Loss", color="red")
    ax1.plot(rounds, losses, color="red", label="Loss")
    ax1.tick_params(axis="y", labelcolor="red")

    # Creating second y-axis for accuracy
    ax2 = ax1.twinx()
    ax2.set_ylabel("Accuracy (%)", color="blue")
    ax2.plot(rounds, accuracies, marker="o", color="blue", label="Accuracy")
    ax2.tick_params(axis="y", labelcolor="blue")

    fig.tight_layout()
    plt.title(f"Last Accuracy: {round(accuracies[len(accuracies)-1], 2)}%")
    plt.legend()
    plt.grid(True)
    plt.savefig(filename)
    plt.close()


def plot_simulation(
    history,
    cfg: DictConfig,
    dir_name="",
):
    losses = extract_metric_data(history.metrics_centralized["loss"])
    accuracies = [
        100.0 * value
        for value in extract_metric_data(history.metrics_centralized["accuracy"])
    ]
    rounds = [i for i in range(len(losses))]

    base = os.path.join(PLOTS_DIR, dir_name)
    os.makedirs(base, exist_ok=True)
    filename = _format_filename(cfg)

    create_single_plot(f"{base}/{filename}", rounds, losses, accuracies)
