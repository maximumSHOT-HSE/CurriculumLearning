import collections

from transformers import Trainer
from transformers import PreTrainedModel
from transformers.trainer_callback import TrainerState
import datasets
import os
import torch
from torch.utils.data import RandomSampler, Sampler, Dataset, DataLoader, SequentialSampler
from typing import Iterator, Optional, Sequence, List, TypeVar, Generic, Sized
import numpy as np
import math
from transformers.file_utils import is_torch_tpu_available
from transformers.trainer_pt_utils import get_tpu_sampler


class SequentialTrainer(Trainer):

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def _get_train_sampler(self) -> Optional[torch.utils.data.sampler.Sampler]:
        if isinstance(self.train_dataset, torch.utils.data.IterableDataset) or not isinstance(
            self.train_dataset, collections.abc.Sized
        ):
            return None
        return SequentialSampler(self.train_dataset)


class ReverseSequentialSampler(Sampler):

    def __init__(self, data_source):
        super().__init__(data_source)
        self.data_source = data_source

    def __iter__(self):
        return iter(range(len(self.data_source) - 1, -1, -1))

    def __len__(self) -> int:
        return len(self.data_source)


class ReverseSequentialTrainer(Trainer):

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def _get_train_sampler(self) -> Optional[torch.utils.data.sampler.Sampler]:
        if isinstance(self.train_dataset, torch.utils.data.IterableDataset) or not isinstance(
            self.train_dataset, collections.abc.Sized
        ):
            return None
        return ReverseSequentialSampler(self.train_dataset)



class CurriculumSamplerHyperbole(Sampler):

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
        super().__init__(*args, **kwargs)
        self.n_bins = n_bins
        self.window_width = window_width
        self.n_see = n_see
        self.ro = ro
        self.drop = drop
        self.drop_ratio = drop_ratio

    def _get_train_sampler(self) -> Optional[torch.utils.data.sampler.Sampler]:
        return CurriculumSamplerHyperbole(
            data_source=self.train_dataset,
            state=self.state,
            n_bins=self.n_bins,
            window_width=self.window_width,
            n_see=self.n_see,
            ro=self.ro,
            drop=self.drop,
            drop_ratio=self.drop_ratio
        )


class CurriculumSamplerDifficultyBiased(Sampler):

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
        super().__init__(*args, **kwargs)
        self.n_bins = n_bins
        self.n_see = n_see

    def _get_train_sampler(self) -> Optional[torch.utils.data.sampler.Sampler]:
        return CurriculumSamplerDifficultyBiased(
            data_source=self.train_dataset,
            state=self.state,
            n_bins=self.n_bins,
            n_see=self.n_see
        )


class CurriculumSamplerCompetenceBased(Sampler):

    def get_sqrt_competence(self):
        return lambda t: min(1, math.sqrt(t * (1 - self.c0 ** 2) / self.T + self.c0 ** 2))

    def get_linear_comptence(self):
        return lambda t: min(1, t * (1 - self.c0) / self.T + self.c0)

    def __init__(
        self,
        data_source: Optional[Sized],
        curriculum_ratio: float,
        max_steps: int,
        batch_size: int,
        c0: float = 0.2,
        type: str = 'sqrt'
    ):
        super().__init__(data_source)
        self.data_source = data_source
        self.curriculum_ratio = curriculum_ratio
        self.max_steps = max_steps
        self.batch_size = batch_size
        assert type in ['sqrt', 'linear']
        self.type = type
        self.c0 = c0
        self.T = int(curriculum_ratio * max_steps)
        self.competence = {
            'sqrt': self.get_sqrt_competence(),
            'linear': self.get_linear_comptence()
        }[self.type]
        self.size = len(self.data_source)
        self.ps = []

    def __iter__(self):
        for t in range(self.max_steps):
            prefix_size = math.ceil(self.competence(t) * self.size)
            prefix_size = max(1, min(prefix_size, self.size))
            ids = np.random.choice(a=prefix_size, size=self.batch_size, replace=True)
            self.ps.append(prefix_size)
            for id in ids:
                yield int(id)

    def __len__(self):
        return self.max_steps * self.batch_size


class CurriculumTrainerCompetenceBased(Trainer):

    def __init__(self, curriculum_ratio=0.2, c0=0.2, type='sqrt', *args, **kwargs):
        super().__init__(*args, **kwargs)
        assert self.args.max_steps > 0
        self.curriculum_ratio = curriculum_ratio
        self.c0 = c0
        self.type = type

    def _get_train_sampler(self) -> Optional[torch.utils.data.sampler.Sampler]:
        return CurriculumSamplerCompetenceBased(
            data_source=self.train_dataset,
            curriculum_ratio=self.curriculum_ratio,
            max_steps=self.args.max_steps,
            batch_size=self.args.train_batch_size,
            c0=self.c0,
            type=self.type
        )


class FromFileSampler(Sampler):

    def __init__(self, data_source, file: str = None):
        super().__init__(data_source)
        self.data_source = data_source
        self.file = file

    def __iter__(self):
        with open(self.file, 'r') as fin:
            for line in fin:
                x = line.strip().strip('\x00')
                yield int(x)

    def __len__(self) -> int:
        return len(self.data_source)


class FromFileTrainer(Trainer):

    def __init__(self, file: str = None, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.file = file

    def _get_train_sampler(self) -> Optional[torch.utils.data.sampler.Sampler]:
        if isinstance(self.train_dataset, torch.utils.data.IterableDataset) or not isinstance(
            self.train_dataset, collections.abc.Sized
        ):
            return None
        return FromFileSampler(self.train_dataset, self.file)


class LadderSampler(Sampler):

    def __init__(
        self,
        data_source,
        curriculum_ratio: float,
        max_steps: int,
        n_stairs: int,
        batch_size: int,
    ):
        super().__init__(data_source)
        self.data_source = data_source
        self.curriculum_ratio = curriculum_ratio
        self.max_steps = max_steps
        self.n_stairs = n_stairs
        self.batch_size = batch_size


        self.ps = []

    def __iter__(self):
        curriculum_length = int(self.curriculum_ratio * self.max_steps)
        stair_length = int(curriculum_length / self.n_stairs)
        for t in range(self.max_steps):
            stair_height = (t // stair_length + 1) / self.n_stairs
            stair_height = np.clip(stair_height, 0, 1)

            prefix_size = int(stair_height * len(self.data_source))
            prefix_size = np.clip(prefix_size, 1, len(self.data_source))

            self.ps.append(prefix_size)

            ids = np.random.choice(a=prefix_size, size=self.batch_size, replace=True).reshape(-1)
            for id in ids:
                yield int(id)

    def __len__(self):
        return self.max_steps * self.batch_size


class LadderTrainer(Trainer):

    def __init__(
        self,
        curriculum_ratio: float = 0.7,
        n_stairs: int = 3,
        *args,
        **kwargs
    ):
        super().__init__(*args, **kwargs)
        assert self.args.max_steps > 0
        self.curriculum_ratio = curriculum_ratio
        self.n_stairs = n_stairs

    def _get_train_sampler(self):
        return LadderSampler(
            data_source=self.train_dataset,
            curriculum_ratio=self.curriculum_ratio,
            max_steps=self.args.max_steps,
            n_stairs=self.n_stairs,
            batch_size=self.args.train_batch_size
        )

