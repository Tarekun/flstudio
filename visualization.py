import os
import matplotlib.pyplot as plt


PLOTS_DIR = "./plots"
os.makedirs(PLOTS_DIR, exist_ok=True)


def extract_metric_data(metric_history):
    return [data[1] for data in metric_history]


def create_plot(filename: str, x, y, title="", ylabel=""):
    # plt.figure(figsize=(8, 6))
    plt.plot(x, y, marker="o")
    plt.title(title)
    # plt.xlabel("Round")
    plt.ylabel(ylabel)
    plt.grid(True)
    plt.xticks(range(0, len(x)))
    plt.savefig(os.path.join(PLOTS_DIR, filename))
    plt.close()


def plot_simulation(history):
    losses = extract_metric_data(history.metrics_centralized["loss"])
    accuracies = [
        100.0 * value
        for value in extract_metric_data(history.metrics_centralized["accuracy"])
    ]
    rounds = [i for i in range(len(losses))]

    create_plot("loss_per_round.png", x=rounds, y=losses, title="Loss per round")
    create_plot(
        "accuracy_per_round.png",
        x=rounds,
        y=accuracies,
        title="Accuracy per round",
        ylabel=f"last: {round(accuracies[len(accuracies)-1], 2)}%",
    )
