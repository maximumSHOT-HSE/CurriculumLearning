import collections

from transformers import Trainer
from transformers.trainer_callback import TrainerState
import datasets
import os
import torch
from torch.utils.data import RandomSampler, Sampler, Dataset, DataLoader
from typing import Iterator, Optional, Sequence, List, TypeVar, Generic, Sized
import numpy as np
import math
from transformers.file_utils import is_torch_tpu_available
from transformers.trainer_pt_utils import get_tpu_sampler
from torch.utils.data.distributed import DistributedSampler


class CurriculumSamplerHyperbole(Sampler):
    r"""Samples elements in the following way:
        1. All dataset is splitted into n_bins bins (last bin may have smaller size than others)
        2. Sampler assumes that will be 1 epoch. 
        3. Sampler gives the probabilities for each bin: for each 0 <= i <= window_width,
            let q be the current main bin
            let t = (q - window_width + 1) be the center of the current window, then
            weight of the i-th bin will be equal to 1 / (|i - t| + 1)^ro
            then sampler will sample indices from given bins with that weights.
            In othe words, we will consider distribution with some center, where located the biggest mass and mass linearly decreases both to the
            right and to the left of the center. Center will lsightly move to the right
        4. Notice that at the end bins will have almost equal
            expected number (n_see) of times when index from i-th bin will be sampled
    """

    def __init__(
        self,
        data_source: Optional[Sized],
        state: TrainerState,
        n_bins: int,
        window_width: int,
        n_see: int,
        ro: float
    ):
        super().__init__(data_source)
        self.data_source = data_source
        self.state = state
        self.n_bins = n_bins
        self.size = len(self.data_source)
        self.window_width = window_width
        self.n_see = n_see
        self.bin_size = math.ceil(self.size / n_bins)
        self.ro = ro

        self.indices = self.build_indices()

    def build_indices(self):
        indices = []
        for t in range(-self.window_width + 1, self.n_bins + self.window_width - 1):
            for _ in range(self.n_see):
                p = 1 / (abs(np.arange(self.n_bins) - t) + 1) ** self.ro
                p /= p.sum()
                ids = np.random.choice(self.n_bins, self.bin_size, p=p) * self.bin_size + np.random.choice(self.bin_size, self.bin_size)
                ids = ids[ids < self.size]
                indices.append(ids)
        return np.concatenate(indices)

    def __iter__(self):
        yield from self.indices

    def __len__(self):
        return len(self.indices)


class CurriculumTrainerHyperbole(Trainer):
    def __init__(self, n_bins=10, window_width=3, n_see=3, ro=0.5, *args, **kwargs):
        super(CurriculumTrainerHyperbole, self).__init__(*args, **kwargs)
        self.n_bins = n_bins
        self.window_width = window_width
        self.n_see = n_see
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
                    n_see=self.n_see,
                    ro=self.ro,
                )
                if self.args.local_rank == -1
                else DistributedSampler(self.train_dataset)
            )
