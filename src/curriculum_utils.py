from transformers import Trainer, TrainingArguments
from transformers import DataCollatorForLanguageModeling
from transformers import LineByLineTextDataset
from transformers import BertTokenizer, BertConfig, BertForMaskedLM
import datasets
import os
import torch
from torch.utils.data import RandomSampler, Sampler, Dataset, DataLoader
from typing import Iterator, Optional, Sequence, List, TypeVar, Generic, Sized
import numpy as np


class CurriculumSampler(Sampler):

    def __init__(self, 
        data_source: Optional[Sized],
    ) -> None:
        super().__init__(data_source)
        self.data_source = data_source

    def __iter__(self):
        pass

    def __len__(self):
        pass


class CurriculumTrainer(Trainer):

    def _get_train_sampler(self) -> Optional[torch.utils.data.sampler.Sampler]:
        if isinstance(self.train_dataset, torch.utils.data.IterableDataset) or not isinstance(
            self.train_dataset, collections.abc.Sized
        ):
            return None
        elif is_torch_tpu_available():
            return get_tpu_sampler(self.train_dataset)
        else:
            return (
                CurriculumSampler()
                if self.args.local_rank == -1
                else DistributedSampler(self.train_dataset)
            )


class MyDataset(Dataset):
    def __init__(self, n):
        self.n = n
        self.x = np.arange(n)

    def __len__(self):
        return self.n

    def __getitem__(self, i):
        return self.x[i]

dataset = MyDataset(1000)
sampler = RandomSampler(dataset)

for x in sampler:
    print(x)


