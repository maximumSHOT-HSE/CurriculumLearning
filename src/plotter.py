import plotly.graph_objects as go
import sys
import json
import subprocess
from pathlib import Path
from argparse import ArgumentParser
import numpy as np
from enum import Enum
from shutil import copyfile
import os

from plotly.subplots import make_subplots

LOG_DIRECTORY = "Logs"


class PathType(Enum):
    LOCAL = 0,
    CLUSTER = 1

    @staticmethod
    def from_name(type_):
        if type_ == 'local':
            return PathType.LOCAL
        elif type_ == 'cluster':
            return PathType.CLUSTER
        else:
            raise


class Experiment:
    def __init__(self, path_type, path, experiment_group):
        self.path_type = PathType.from_name(path_type)
        self.path = path
        self.experiment_group = experiment_group


def load_config(path):
    with open(path) as f:
        config = json.load(f)

    return config


def load_from_cluster(local_path, cluster_path):
    subprocess.run(['scp', '-rP', '2222', f'aomelchenko@cluster.hpc.hse.ru:{cluster_path}', local_path])


def get_last_checkpoint(experiment):
    if experiment.path_type == PathType.CLUSTER:
        checkpoints = subprocess.Popen(['ssh', '-p', '2222', f'aomelchenko@cluster.hpc.hse.ru', f'ls {experiment.path}'],
                                       stdout=subprocess.PIPE).stdout.read().decode("utf-8").strip().split('\n')
    else:
        checkpoints = os.listdir(experiment.path)

    return sorted(checkpoints, key=lambda checkpoint: int(checkpoint.split('-')[1]))[-1]


def load_all_experiments(all_experiments):
    for experiment_group in all_experiments.keys():
        for i, experiment in enumerate(all_experiments[experiment_group]):
            local_path = f'{LOG_DIRECTORY}/{experiment.experiment_group}{i}.json'
            previous_path = f'{experiment.path}/{get_last_checkpoint(experiment)}/trainer_state.json'
            if experiment.path_type == PathType.CLUSTER:
                load_from_cluster(local_path=local_path, cluster_path=previous_path)
            else:
                copyfile(previous_path, local_path)


def get_loss_history(history, metric):
    metrics = []
    epochs = []

    for log in history:
        if metric in log:
            metrics.append(log[metric])
            epochs.append(log['step'])

    return epochs, metrics


def get_experiment_data(experiments, title):
    all_losses = []
    all_epochs = []

    for i, experiment in enumerate(experiments):
        path_to_config = f'{LOG_DIRECTORY}/{experiment.experiment_group}{i}.json'
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
    plot_titles = ['eval_accuracy', 'eval_f1', 'eval_precision', 'eval_recall', 'eval_loss', 'loss']

    fig = make_subplots(rows=3, cols=2, subplot_titles=plot_titles)

    for num, experiment_group in enumerate(experiments.keys()):
        for i, title in enumerate(plot_titles):
            epochs, values = get_experiment_data(experiments[experiment_group], title)
            row = i // 2 + 1
            col = i % 2 + 1

            means = values.mean(axis=0)
            deviations = values.std(axis=0)

            values_lower = (means - deviations).tolist()[::-1]
            values_upper = (means + deviations).tolist()

            fig.add_trace(go.Scatter(x=epochs, y=means, name=experiment_group,
                                     legendgroup=experiment_group, showlegend=i == 0,
                                     line=dict(color=COLORS[num])),
                          row=row, col=col)
            fig.add_trace(go.Scatter(x=epochs + epochs[::-1], y=values_upper + values_lower, fill='tozerox',
                                     fillcolor=FILL_COLORS[num], line=dict(color='rgba(255,255,255,0)'), opacity=0,
                                     showlegend=False, legendgroup=experiment_group, name=experiment_group),
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
    all_experiments = {}
    with open(config, 'r') as f:
        for experiment_string in [line.strip() for line in f.readlines()]:
            path_type, path, experiment_group = experiment_string.split(' ')
            experiment = Experiment(path_type, path, experiment_group)

            if experiment.experiment_group not in all_experiments:
                all_experiments[experiment_group] = []
            all_experiments[experiment_group].append(experiment)

    return all_experiments


if __name__ == '__main__':
    show_results()
