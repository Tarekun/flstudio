import os
import matplotlib.pyplot as plt


PLOTS_DIR = "./plots"
os.makedirs(PLOTS_DIR, exist_ok=True)


def extract_metric_data(metric_history):
    return [data[1] for data in metric_history]


def _format_filename(
    num_clients: int = None, lr: float = None, hybrid_ratio: float = None
):
    name = "results"
    if not num_clients is None:
        name += f"-c{num_clients}"
    if not hybrid_ratio is None:
        name += f"-h{hybrid_ratio}"
    if not lr is None:
        name += f"-lr{lr}"

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
    dir_name="",
    # optional arguments to be set only if they are meaningful to the simulation ran
    # they are used as naming conventions for the plot file
    num_clients: int = None,
    lr: float = None,
    hybrid_ratio: float = None,
):
    losses = extract_metric_data(history.metrics_centralized["loss"])
    accuracies = [
        100.0 * value
        for value in extract_metric_data(history.metrics_centralized["accuracy"])
    ]
    rounds = [i for i in range(len(losses))]

    base = os.path.join(PLOTS_DIR, dir_name)
    os.makedirs(base, exist_ok=True)
    filename = _format_filename(
        num_clients=num_clients, lr=lr, hybrid_ratio=hybrid_ratio
    )

    create_single_plot(f"{base}/{filename}", rounds, losses, accuracies)
