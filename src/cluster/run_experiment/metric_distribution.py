import numpy as np
from plotly import graph_objects as go
import os
from tqdm import tqdm
from argparse import ArgumentParser
from datasets import load_from_disk


def parse_arguments():
    parser = ArgumentParser()
    parser.add_argument('--dataset', type=str,
                        required=True)
    parser.add_argument('--metric', type=str, default='tse')
    parser.add_argument('--output', type=str, default='statistics.html')
    parser.add_argument('--clip_max', type=float, default=-1)
    parser.add_argument('--max_bins', type=int, default=1000)

    return parser.parse_args()


def load_experiment(dataset_path, metric_name):
    ds = load_from_disk(dataset_path)
    metrics = []
    for part in ds.keys():
        metrics.append(np.array(ds[part][metric_name]))

    return np.hstack(metrics)


def show_histogram(dataset_path, metric_name, output_path, clip_max, max_bins):
    fig = go.Figure()

    metrics = load_experiment(dataset_path, metric_name)
    if clip_max != -1:
        metrics = metrics.clip(0, clip_max)
    fig.add_trace(go.Histogram(x=metrics, nbinsx=max_bins))

    fig.update_layout(
        title_text=f'Metric: {metric_name.upper()}',
        xaxis_title_text='Metric',
        yaxis_title_text='Count'
    )

    fig.write_html(output_path)


if __name__ == '__main__':
    args = parse_arguments()
    show_histogram(dataset_path=args.dataset, metric_name=args.metric,
                   output_path=args.output, clip_max=args.clip_max,
                   max_bins=args.max_bins)
