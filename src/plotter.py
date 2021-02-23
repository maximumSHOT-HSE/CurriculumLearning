import plotly.graph_objects as go
import sys
import json
import subprocess
from pathlib import Path
from argparse import ArgumentParser
import numpy as np

from plotly.subplots import make_subplots

LOG_DIRECTORY = "Logs"


def load_config(path):
    with open(path) as f:
        config = json.load(f)

    return config


def get_short_names(experiments):
    splitted_experiments = []
    for experiment in experiments:
        splitted_experiments.append([path.split('/') for path in experiment])

    for i in range(len(splitted_experiments[0][0])):
        if splitted_experiments[0][0][i] != splitted_experiments[1][0][i]:
            return [[f'{experiment[j][i]}_{j}' for j in range(len(experiment))] for experiment in splitted_experiments]


def load_from_cluster(local_path, cluster_path):
    subprocess.run(['scp', '-rP', '2222', f'aomelchenko@cluster.hpc.hse.ru:{cluster_path}', local_path])


def get_last_checkpoint(experiment_path):
    checkpoints = subprocess.Popen(['ssh', '-p', '2222', f'aomelchenko@cluster.hpc.hse.ru',
                                    f'ls {experiment_path}'],
                                   stdout=subprocess.PIPE).stdout.read().decode("utf-8").strip().split('\n')

    return sorted(checkpoints, key=lambda checkpoint: int(checkpoint.split('-')[1]))[-1]


def load_all_experiments(all_experiments):
    all_short_names = get_short_names(all_experiments)
    for short_names, experiment_paths in zip(all_short_names, all_experiments):
        for short_name, experiment_path in zip(short_names, experiment_paths):
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


def get_experiment_data(experiment_names, title):
    all_losses = []
    all_epochs = []

    for experiment_name in experiment_names:
        path_to_config = f'{LOG_DIRECTORY}/{experiment_name}.json'
        history = load_config(path_to_config)['log_history']
        epochs, losses = get_loss_history(history, title)
        all_losses.append(losses)
        all_epochs.append(epochs)

    min_epoch = min([len(epochs) for epochs in all_epochs])
    cut_losses = [losses[:min_epoch] for losses in all_losses]

    return all_epochs[0][:min_epoch], np.array(cut_losses)


COLORS = ['rgb(31, 119, 180)', 'rgb(255, 127, 14)', 'rgb(44, 160, 44)', 'rgb(214, 39, 40)', 'rgb(148, 103, 189)',
          'rgb(140, 86, 75)', 'rgb(227, 119, 194)', 'rgb(127, 127, 127)']

FILL_COLORS = ['rgba(31, 119, 180, 0.2)', 'rgba(255, 127, 14, 0.2)', 'rgba(44, 160, 44, 0.2)', 'rgba(214, 39, 40, 0.2)',
               'rgba(148, 103, 189, 0.2)', 'rgba(140, 86, 75, 0.2)', 'rgba(227, 119, 194, 0.2)', 'rgba(127, 127, 127, 0.2)']


def plot(experiments):
    short_names = get_short_names(experiments)
    plot_titles = ['eval_accuracy', 'eval_f1', 'eval_precision', 'eval_recall', 'eval_loss', 'loss']

    fig = make_subplots(rows=3, cols=2, subplot_titles=plot_titles)

    for num, experiment in enumerate(short_names):
        for i, title in enumerate(plot_titles):
            epochs, values = get_experiment_data(experiment, title)
            row = i // 2 + 1
            col = i % 2 + 1

            means = values.mean(axis=0)
            deviations = values.std(axis=0)

            values_lower = (means - deviations).tolist()[::-1]
            values_upper = (means + deviations).tolist()

            fig.add_trace(go.Scatter(x=epochs, y=means, name=experiment[0][:-2], legendgroup=experiment[0][:-2], showlegend=i == 0,
                                     line=dict(color=COLORS[num])),
                          row=row, col=col)
            fig.add_trace(go.Scatter(x=epochs + epochs[::-1], y=values_upper + values_lower, fill='tozerox',
                                     fillcolor=FILL_COLORS[num], line=dict(color='rgba(255,255,255,0)'), opacity=0,
                                     showlegend=False, legendgroup=experiment[0][:-2], name=experiment[0][:-2]),
                          row=row, col=col)
            fig.update_xaxes(title_text="steps", row=row, col=col)
            fig.update_yaxes(title_text=title, row=row, col=col)

    fig.write_html("results.html")
    #fig.show()


def show_results():
    config_path = parse_config()
    Path(LOG_DIRECTORY).mkdir(parents=True, exist_ok=True)
    experiments = load_experiment_paths(config_path)
    load_all_experiments(experiments)
    plot(experiments)


def parse_config():
    parser = ArgumentParser()
    parser.add_argument("--config", type=str, default="config.txt")
    parsed_args = parser.parse_args()

    return parsed_args.config


def load_experiment_paths(config):
    all_experiments = []
    with open(config, 'r') as f:
        single_experiments = []
        for path in [line.strip() for line in f.readlines()]:
            if path == "":
                if len(single_experiments) > 0:
                    all_experiments.append(single_experiments)
                single_experiments = []
            else:
                single_experiments.append(path)

        if len(single_experiments) > 0:
            all_experiments.append(single_experiments)

    return all_experiments

if __name__ == '__main__':
    show_results()
