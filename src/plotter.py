import plotly.graph_objects as go
import sys
import json
import subprocess
from pathlib import Path


LOG_DIRECTORY = "BertLogs"


def load_config(path):
    with open(path) as f:
        config = json.load(f)

    return config


EXPERIMENTS = ['base', 'ee', 'tse', 'tse_difficult', 'len', 'tse_div_len']


def load_from_cluster(local_path, cluster_path):
    subprocess.run(['scp', '-rP', '2222', f'aomelchenko@cluster.hpc.hse.ru:{cluster_path}', local_path])


def get_last_checkpoint(experiment_name):
    checkpoints = subprocess.Popen(['ssh', '-p', '2222', f'aomelchenko@cluster.hpc.hse.ru',
                                    f'ls Bachelor-s-Degree/Logs/{experiment_name}_bert'],
                                   stdout=subprocess.PIPE).stdout.read().decode("utf-8").strip().split('\n')

    return sorted(checkpoints, key=lambda checkpoint: int(checkpoint.split('-')[1]))[-1]


def load_all_experiments():
    for experiment_name in EXPERIMENTS:
        cluster_path = f'Bachelor-s-Degree/Logs/{experiment_name}_bert/' \
                       f'{get_last_checkpoint(experiment_name)}/trainer_state.json'
        local_path = f'{LOG_DIRECTORY}/{experiment_name}.json'
        load_from_cluster(local_path=local_path, cluster_path=cluster_path)


def get_loss(log):
    if 'eval_loss' in log:
        return log['eval_loss']
    else:
        return log['loss']


def get_loss_history(history, is_eval=True):
    losses = []
    epochs = []

    for log in history:
        eval_loss = 'eval_loss' in log
        if eval_loss == is_eval:
            losses.append(get_loss(log))
            epochs.append(log['step'])

    return epochs, losses


def get_experiment_data(experiment_name):
    path_to_config = f'{LOG_DIRECTORY}/{experiment_name}.json'
    history = load_config(path_to_config)['log_history']

    epochs, losses = get_loss_history(history)
    return epochs, losses


def plot():
    fig = go.Figure()

    for experiment in EXPERIMENTS:
        epochs, losses = get_experiment_data(experiment)
        fig.add_trace(go.Scatter(x=epochs, y=losses, name=experiment))

    fig.update_layout(title='Bert Losses',
                      xaxis_title='Epoch',
                      yaxis_title='Loss')

    fig.show()


def show_results():
    Path(LOG_DIRECTORY).mkdir(parents=True, exist_ok=True)
    load_all_experiments()
    plot()


if __name__ == '__main__':
    show_results()
