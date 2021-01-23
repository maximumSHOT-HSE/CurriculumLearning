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
        ro: float,
        drop: bool = False,
        drop_ratio: float = 0.1
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
        self.drop = drop
        self.drop_ratio = drop_ratio
        self.indices = self.build_indices()

    def build_indices(self):
        indices = []
        for t in range(-self.window_width + 1, self.n_bins + self.window_width - 1):
            for _ in range(self.n_see):
                p = 1 / (abs(np.arange(self.n_bins) - t) + 1) ** self.ro
                p /= p.sum()
                k = math.ceil(self.size / (self.n_bins + 2 * self.window_width - 2))
                ids = np.random.choice(self.n_bins, k, p=p) * self.bin_size + np.random.choice(self.bin_size, k)
                ids = ids[ids < self.size]
                indices.append(ids)
        result = np.concatenate(indices).tolist()
        if self.drop:
            drop_size = int(self.drop_ratio * self.size)
            result = list(filter(lambda i: i > drop_size, result))
        return result

    def __iter__(self):
        yield from self.indices

    def __len__(self):
        return len(self.indices)


class CurriculumTrainerHyperbole(Trainer):

    def __init__(self, n_bins=10, window_width=3, n_see=3, ro=0.5, drop=False, drop_ratio=0.1, *args, **kwargs):
        super(CurriculumTrainerHyperbole, self).__init__(*args, **kwargs)
        self.n_bins = n_bins
        self.window_width = window_width
        self.n_see = n_see
        self.ro = ro
        self.drop = drop
        self.drop_ratio = drop_ratio

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
                    drop=drop,
                    drop_ratio = drop_ratio
                )
                if self.args.local_rank == -1
                else DistributedSampler(self.train_dataset)
            )


class CurriculumSamplerDifficultyBiased(Sampler):
    r"""Samples elements in the following way:
        Samplers goes throught bins. Suppose that we are on the i-th bin.
        Sampler samples (n - i) times the batch of size k=n_see * size * 2 / (n_bins * (n_bins - 1)).
        The total number of samples is equalt to n_see * size and the final distribution is the following:
        i-th bin will have weight w_i = i (the weight increasing from simple bins to difficult bins).
    """

    def __init__(
        self,
        data_source: Optional[Sized],
        state: TrainerState,
        n_bins: int,
        n_see: int
    ):
        super().__init__(data_source)
        self.data_source = data_source
        self.state = state
        self.n_bins = n_bins
        self.n_see = n_see
        self.size = len(self.data_source)
        self.bin_size = math.ceil(self.size / n_bins)
        
        self.indices = self.build_indices()

    def build_indices(self):
        indices = []
        k = math.ceil(self.n_see * self.size * 2 / self.n_bins / (self.n_bins + 1))
        for t in range(self.n_bins):
            for _ in range(self.n_bins - t):
                p = np.zeros(self.n_bins)
                p[t:] = 1
                p /= p.sum()
                ids = np.random.choice(self.n_bins, k, p=p) * self.bin_size + np.random.choice(self.bin_size, k)
                ids = ids[ids < self.size]
                indices.append(ids)
        return np.concatenate(indices).tolist()

    def __iter__(self):
        yield from self.indices

    def __len__(self):
        return len(self.indices)


class CurriculumTrainerDifficultyBiased(Trainer):

    def __init__(self, n_bins=10, n_see=3, *args, **kwargs):
        super(CurriculumTrainerDifficultyBiased, self).__init__(*args, **kwargs)
        self.n_bins = n_bins
        self.n_see = n_see

    def _get_train_sampler(self) -> Optional[torch.utils.data.sampler.Sampler]:
        if isinstance(self.train_dataset, torch.utils.data.IterableDataset) or not isinstance(
            self.train_dataset, collections.abc.Sized
        ):
            return None
        elif is_torch_tpu_available():
            return get_tpu_sampler(self.train_dataset)
        else:
            return (
                CurriculumSamplerDifficultyBiased(
                    data_source=self.train_dataset,
                    state=self.state,
                    n_bins=self.n_bins,
                    n_see=self.n_see
                )
                if self.args.local_rank == -1
                else DistributedSampler(self.train_dataset)
            )


class CurriculumSamplerCompetenceBased(Sampler):

    def get_sqrt_competence(self):
        return lambda t: min(1, math.sqrt(t * (1 - self.c0 ** 2) / self.T + self.c0 ** 2))

    def get_linear_comptence(self):
        return lambda t: min(1, t * (1 - self.c0) / self.T + self.c0)

    def __init__(
        self, 
        data_source: Optional[Sized],
        state: TrainerState,
        n_see: int,
        batch_size: int,
        c0: float = 0.01,
        type: str = 'sqrt'
    ):
        super().__init__(data_source)
        self.data_source = data_source
        self.size = len(data_source)
        self.state = state
        self.n_see = n_see
        self.batch_size = batch_size
        assert type in ['sqrt', 'linear']
        self.type = type
        self.T = math.ceil(self.n_see * self.size / self.batch_size)
        self.c0 = c0
        self.competence = {
            'sqrt': self.get_sqrt_competence(),
            'linear': self.get_linear_comptence()
        }[self.type]

        self.ps = []
        self.indices = self.build_indices()

    def build_indices(self):
        indices = []
        for t in range(0, self.T):
            prefix_size = math.ceil(self.competence(t) * self.size)
            prefix_size = max(1, min(prefix_size, self.size))
            ids = np.random.choice(a=prefix_size, size=self.batch_size, replace=True)
            indices.append(ids)
            self.ps.append(prefix_size)
        return np.concatenate(indices).tolist()       

    def __iter__(self):
        yield from self.indices

    def __len__(self):
        return len(self.indices) 


class CurriculumTrainerCompetenceBased(Trainer):

    def __init__(self, n_see=3, batch_size=128, c0=0.01, type='sqrt', *args, **kwargs):
        super(CurriculumTrainerDifficultyBiased, self).__init__(*args, **kwargs)
        self.n_see = n_see,
        self.batch_size = batch_size,
        self.c0 = c0
        self.type = type

    def _get_train_sampler(self) -> Optional[torch.utils.data.sampler.Sampler]:
        if isinstance(self.train_dataset, torch.utils.data.IterableDataset) or not isinstance(
            self.train_dataset, collections.abc.Sized
        ):
            return None
        elif is_torch_tpu_available():
            return get_tpu_sampler(self.train_dataset)
        else:
            return (
                CurriculumSamplerCompetenceBased(
                    data_source=self.train_dataset,
                    state=self.state,
                    n_see=self.n_see,
                    batch_size=self.batch_size,
                    c0=self.c0,
                    type=self.type
                )
                if self.args.local_rank == -1
                else DistributedSampler(self.train_dataset)
            )
