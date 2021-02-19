import plotly.graph_objects as go
import sys
import json
import subprocess
from pathlib import Path
from argparse import ArgumentParser

from plotly.subplots import make_subplots

LOG_DIRECTORY = "Logs"


def load_config(path):
    with open(path) as f:
        config = json.load(f)

    return config


def get_short_names(experiments):
    splitted_experiments = []
    for experiment in experiments:
        splitted_experiments.append(experiment.split('/'))

    for i in range(len(splitted_experiments[0])):
        if splitted_experiments[0][i] != splitted_experiments[1][i]:
            return [splitted_experiments[j][i] for j in range(len(splitted_experiments))]


def load_from_cluster(local_path, cluster_path):
    subprocess.run(['scp', '-rP', '2222', f'aomelchenko@cluster.hpc.hse.ru:{cluster_path}', local_path])


def get_last_checkpoint(experiment_path):
    checkpoints = subprocess.Popen(['ssh', '-p', '2222', f'aomelchenko@cluster.hpc.hse.ru',
                                    f'ls {experiment_path}'],
                                   stdout=subprocess.PIPE).stdout.read().decode("utf-8").strip().split('\n')

    return sorted(checkpoints, key=lambda checkpoint: int(checkpoint.split('-')[1]))[-1]


def load_all_experiments(experiments):
    short_names = get_short_names(experiments)
    for short_name, experiment_path in zip(short_names, experiments):
        cluster_path = f'{experiment_path}/{get_last_checkpoint(experiment_path)}/trainer_state.json'
        local_path = f'{LOG_DIRECTORY}/{short_name}.json'
        print(cluster_path)
        load_from_cluster(local_path=local_path, cluster_path=cluster_path)


def get_loss_history(history, metric):
    metrics = []
    epochs = []

    for log in history:
        if metric in log:
            metrics.append(log[metric])
            epochs.append(log['step'])

    return epochs, metrics


def get_experiment_data(experiment_name, title):
    path_to_config = f'{LOG_DIRECTORY}/{experiment_name}.json'
    history = load_config(path_to_config)['log_history']

    epochs, losses = get_loss_history(history, title)
    return epochs, losses


def plot(experiments):
    short_names = get_short_names(experiments)
    plot_titles = ['eval_accuracy', 'eval_f1', 'eval_precision', 'eval_recall', 'eval_loss', 'loss']

    fig = make_subplots(rows=3, cols=2, subplot_titles=plot_titles)

    for experiment in short_names:
        for i, title in enumerate(plot_titles):
            epochs, values = get_experiment_data(experiment, title)
            row = i // 2 + 1
            col = i % 2 + 1
            fig.add_trace(go.Scatter(x=epochs, y=values, name=experiment), row=row, col=col)
            fig.update_xaxes(title_text="steps", row=row, col=col)
            fig.update_yaxes(title_text=title, row=row, col=col)

    fig.write_html("results.html")
    #fig.show()


def show_results():
    config_path = parse_config()
    Path(LOG_DIRECTORY).mkdir(parents=True, exist_ok=True)
    experiments = load_experiment_paths(config_path)
    #load_all_experiments(experiments)
    plot(experiments)


def parse_config():
    parser = ArgumentParser()
    parser.add_argument("--config", type=str, default="config.txt")
    parsed_args = parser.parse_args()

    return parsed_args.config


def load_experiment_paths(config):
    with open(config, 'r') as f:
        return [line.strip() for line in f.readlines()]


if __name__ == '__main__':
    show_results()
