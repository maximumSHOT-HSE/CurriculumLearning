from transformers import Trainer
from transformers.trainer_callback import TrainerState
import datasets
import os
import torch
from torch.utils.data import RandomSampler, Sampler, Dataset, DataLoader
from typing import Iterator, Optional, Sequence, List, TypeVar, Generic, Sized
import numpy as np
import math


class CurriculumSampler(Sampler):
    r"""Samples elements in the following way:
        1. All dataset is splitted into n_bins bins (last bin may have smaller size than others)
        2. At the (q * (n_bins + 2 * window_width - 2) + mod)-th epoch sampler gives the probabilities for each bin: for each 0 <= i <= window_width,
            let t = (q - window_width + 1) be the center of the current window, then
            weight of the (t - window_width + i)-th and (t + window_width - i) bins will be equal to i / (window_width ** 2)
            then sampler will sample indices from given bins with that weights.
            In othe words, we will consider distribution with some center, where located the biggest mass and mass linearly decreases both to the
            right and to the left of the center. Center will move to the right (on cycle).
        3. Notice that end bins will have almost equal expected number of times when index from i-th bin will be sampled
    """

    def __init__(
        self,
        data_source: Optional[Sized],
        state: TrainerState,
        n_bins: int,
        window_width: int,
    ):
        super().__init__(data_source)
        self.data_source = data_source
        self.state = state
        self.n_bins = n_bins
        self.size = len(self.data_source)
        self.window_width = window_width
        self.bin_size = math.ceil(self.size / n_bins)

        self.indices = self.build_indices()

    def build_indices(self):
        p = np.zeros(self.n_bins + 1)
        t = math.floor(self.state.epoch) % self.n_bins  - self.window_width + 1
        for i in range(0, self.window_width + 1):
            for id in [t - self.window_width + i, t + self.window_width - i]:
                if 0 <= id < self.n_bins:
                    p[id] = i
        p /= p.sum()
        p[self.n_bins] = 1 - p.sum()
        ids = np.random.choice(self.n_bins + 1, self.bin_size, p=p) * self.bin_size + np.random.choice(self.bin_size, self.bin_size)
        ids = ids[ids < self.size]
        return ids

    def __iter__(self):
        yield from self.indices

    def __len__(self):
        return len(self.indices)


class CurriculumSamplerHyperbole(Sampler):
    r"""Samples elements in the following way:
        1. All dataset is splitted into n_bins bins (last bin may have smaller size than others)
        2. At the (q * (n_bins + 2 * window_width - 2) + mod)-th epoch sampler gives the probabilities for each bin: for each 0 <= i <= window_width,
            let t = (q - window_width + 1) be the center of the current window, then
            weight of the i-th bins will be 1 / (|i - t| + 1)^ro
            then sampler will sample indices from given bins with that weights.
            In othe words, we will consider distribution with some center, where located the biggest mass and mass linearly decreases both to the
            right and to the left of the center. Center will move to the right.
        3. Notice that at the end bins will have almost equal expected number of times when index from i-th bin will be sampled
    """

    def __init__(
        self,
        data_source: Optional[Sized],
        state: TrainerState,
        n_bins: int,
        window_width: int,
        ro: float
    ):
        super().__init__(data_source)
        self.data_source = data_source
        self.state = state
        self.n_bins = n_bins
        self.size = len(self.data_source)
        self.window_width = window_width
        self.bin_size = math.ceil(self.size / n_bins)
        self.ro = ro

        self.indices = self.build_indices()

    def build_indices(self):
        t = math.floor(self.state.epoch) % self.n_bins - self.window_width + 1
        p = 1 / (abs(np.arange(self.n_bins) - t) + 1) ** self.ro
        p /= p.sum()
        ids = np.random.choice(self.n_bins, self.bin_size, p=p) * self.bin_size + np.random.choice(self.bin_size, self.bin_size)
        ids = ids[ids < self.size]
        return ids

    def __iter__(self):
        yield from self.indices

    def __len__(self):
        return len(self.indices)


class CurriculumTrainer(Trainer):

    def __init__(
        self,
        n_bins: int,
        window_width: int,
        *args,
        **kwargs,
    ):
        super().__init__(*args, **kwargs)
        self.n_bins = n_bins
        self.window_width = window_width

    def _get_train_sampler(self) -> Optional[torch.utils.data.sampler.Sampler]:
        if isinstance(self.train_dataset, torch.utils.data.IterableDataset) or not isinstance(
            self.train_dataset, collections.abc.Sized
        ):
            return None
        elif is_torch_tpu_available():
            return get_tpu_sampler(self.train_dataset)
        else:
            return (
                CurriculumSampler(
                    data_source=self.train_dataset,
                    state=self.state,
                    n_bins=self.n_bins,
                    window_width=self.window_width,
                )
                if self.args.local_rank == -1
                else DistributedSampler(self.train_dataset)
            )


class CurriculumTrainerHyperbole(Trainer):

    def __init__(
        self,
        n_bins: int,
        window_width: int,
        ro: float,
        *args,
        **kwargs,
    ):
        super().__init__(*args, **kwargs)
        self.n_bins = n_bins
        self.window_width = window_width
        self.ro = ro

    def _get_train_sampler(self) -> Optional[torch.utils.data.sampler.Sampler]:
        if isinstance(self.train_dataset, torch.utils.data.IterableDataset) or not isinstance(
            self.train_dataset, collections.abc.Sized
        ):
            return None
        elif is_torch_tpu_available():
            return get_tpu_sampler(self.train_dataset)
        else:
            return (
                CurriculumSamplerHyperbole(
                    data_source=self.train_dataset,
                    state=self.state,
                    n_bins=self.n_bins,
                    window_width=self.window_width,
                    ro=self.ro,
                )
                if self.args.local_rank == -1
                else DistributedSampler(self.train_dataset)
            )
