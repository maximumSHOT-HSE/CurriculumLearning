from transformers import Trainer
from transformers.trainer_callback import TrainerState
import datasets
import os
import torch
from torch.utils.data import RandomSampler, Sampler, Dataset, DataLoader
from typing import Iterator, Optional, Sequence, List, TypeVar, Generic, Sized
import numpy as np
import math
from curriculum_utils import CurriculumSampler, CurriculumSamplerHyperbole
import matplotlib.pyplot as plt
from collections import Counter


class MyDataset(Dataset):
    def __init__(self, n):
        self.n = n
        self.x = np.arange(n)

    def __len__(self):
        return self.n

    def __getitem__(self, i):
        return self.x[i]


def show_hist(dataset_size: int, n_bins: int, window_width: int, n_see: int):
    num_train_epochs=n_see * (n_bins + 2 * window_width - 2)
    total_samples = []

    for epoch in range(0, num_train_epochs):
        state = TrainerState(num_train_epochs=num_train_epochs, epoch=epoch)
        dataset = MyDataset(dataset_size)
        sampler = CurriculumSampler(dataset, state, n_bins, window_width, n_see)

        samples = list(sampler)
        total_samples += samples
        
        print(len(samples))
        continue

        plt.cla()
        plt.clf()
        plt.title(f'Number of views. epoch #{epoch}')
        plt.ylim([0, math.ceil(dataset_size / n_bins)])
        plt.xlabel(f'samples (indices in sorted dataset). n_bins={n_bins}, window_width={window_width}')
        plt.ylabel('number')
        plt.hist(samples, list(range(0, dataset_size, math.ceil(dataset_size / n_bins))) + [dataset_size])
        plt.show()
        # plt.savefig(f'movie/hist_{epoch:03d}.png')

    print()
    print(len(total_samples))

    values = Counter(total_samples).values()
    print(sum(values) / len(values))

    plt.hist(total_samples, bins=1000)
    plt.show()    


def show_hist_hyperbole(dataset_size: int, num_train_epochs: int, n_see: int, n_bins: int, window_width: int, ro, id=-1):
    total_samples = []


    state = TrainerState(num_train_epochs=num_train_epochs, epoch=0)
    dataset = MyDataset(dataset_size)
    sampler = CurriculumSamplerHyperbole(dataset, state, n_bins, window_width, n_see, ro)

    samples = list(sampler)
    k = len(dataset) // n_bins

    step = 0

    for i in range(0, len(samples), k):
        plt.cla()
        plt.clf()
        plt.title(f'Number of views. step #{i}')
        # plt.title(f'ro = {ro}')
        # plt.ylim([0, math.ceil(dataset_size / n_bins)])
        # plt.ylim([0, 500])
        plt.xlabel(f'samples (indices in sorted dataset). n_bins={n_bins}, window_width={window_width}, ro={ro}')
        plt.ylabel('number')
        plt.hist(samples[i: i + k], list(range(0, dataset_size, math.ceil(dataset_size / n_bins))) + [dataset_size])
        # plt.show()
        plt.savefig(f'movie/hist_{step:03d}.png')

        step += 1

    # plt.cla()
    # plt.clf()
    # plt.title(f'Final number of views. ro = {ro}')
    # # plt.hist(total_samples, list(range(0, dataset_size, math.ceil(dataset_size / n_bins))) + [dataset_size])
    # plt.hist(samples, bins=10000)
    # plt.show()
    # # plt.savefig(f'movie/hist_{id:03d}.png')

    counter = Counter(samples)
    values = counter.values()

    print(min(values), max(values), sum(values) / len(values))


if __name__ == '__main__':
    # show_hist(
    #     dataset_size=100000,
    #     n_bins=50,
    #     window_width=8,
    #     n_see=5
    # )
    show_hist_hyperbole(
        dataset_size=100000,
        n_see=3,
        num_train_epochs=1,
        n_bins=50,
        window_width=8,
        ro=0.5
    )

    # ros = np.linspace(0.05, 5, 50)
    # for i, ro in enumerate(ros):
    #     show_hist_hyperbole(
    #         dataset_size=100000,
    #         n_bins=50,
    #         window_width=8,
    #         n_see=5,
    #         ro=ro,
    #         id=i
    #     )
