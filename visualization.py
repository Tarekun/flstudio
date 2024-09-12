import os
import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator
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


def _legend_text(cfg: DictConfig):
    legend = ""
    bias_factor = cfg.data_cfg.get("bias_factor", 0.0)
    legend += f"#clients: {cfg.data_cfg.num_clients}\n"
    legend += f"HR: {cfg.data_cfg.hybrid_ratio}\n"
    legend += f"method: {cfg.data_cfg.hybrid_method}\n"
    legend += f"b: {bias_factor}\n"
    legend += f"lr: {cfg.train_cfg.optimizer.lr}\n"
    legend += f"#epochs: {cfg.train_cfg.epochs}\n"
    legend += f"optim: {cfg.train_cfg.optimizer._target_}\n"
    return legend


def create_single_plot(filename: str, rounds, losses, accuracies, cfg: DictConfig):
    fig, ax1 = plt.subplots()

    # Plotting losses
    ax1.set_xlabel("Round")
    ax1.set_ylabel("Loss", color="red")
    ax1.plot(rounds, losses, color="red")
    ax1.tick_params(axis="y", labelcolor="red")

    # Creating second y-axis for accuracy
    ax2 = ax1.twinx()
    ax2.set_ylabel("Accuracy (%)", color="blue")
    ax2.plot(rounds, accuracies, marker="o", color="blue")
    ax2.tick_params(axis="y", labelcolor="blue")

    ax2.yaxis.set_major_locator(MaxNLocator(integer=True, min_n_ticks=1))
    plt.title(f"Last Accuracy: {round(accuracies[-1], 2)}%")
    fig.tight_layout()
    legend = _legend_text(cfg)
    fig.text(
        0.5,
        -0.25,
        legend,
        ha="center",
        fontsize=10,
        bbox=dict(facecolor="white", alpha=0.5),
    )

    plt.grid(True)
    plt.savefig(filename, bbox_inches="tight")
    plt.close()


def average_metrics(num_rounds: int, histories):
    loss_sum = [0.0] * num_rounds
    accuracy_sum = [0.0] * num_rounds
    num_histories = len(histories)

    for history in histories:
        loss_list = extract_metric_data(history.metrics_centralized["loss"])
        accuracy_list = extract_metric_data(history.metrics_centralized["accuracy"])
        print(accuracy_list)

        # add the values to the corresponding round in the sum lists
        for i in range(num_rounds):
            loss_sum[i] += loss_list[i]
            accuracy_sum[i] += accuracy_list[i]

    # compute the average for each round by dividing the sums by the number of histories
    avg_loss = [loss_sum[i] / num_histories for i in range(num_rounds)]
    avg_accuracy = [accuracy_sum[i] / num_histories for i in range(num_rounds)]
    return avg_loss, avg_accuracy


def plot_simulations(
    histories,
    cfg: DictConfig,
    dir_name="",
):
    losses, accuracies = average_metrics(cfg.sim_cfg.num_rounds, histories)
    accuracies = [100.0 * value for value in accuracies]
    rounds = [i for i in range(len(losses))]

    base = os.path.join(PLOTS_DIR, dir_name)
    os.makedirs(base, exist_ok=True)
    filename = _format_filename(cfg)

    create_single_plot(f"{base}/{filename}", rounds, losses, accuracies, cfg)
