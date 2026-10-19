"""Equalize example counts while retaining every training row once per epoch."""
from torch.utils.data import Sampler
import torch


class EpochBudgetSampler(Sampler):
    """Use common shuffled slots, filling extra slots with original-row repeats.

    Dedicated padding RNG keeps shared rows in the same slots in paired datasets
    with an identical prefix. No truncation or replacement of unique rows occurs.
    """
    def __init__(self, size, budget, seed):
        if size <= 0 or budget < size:
            raise ValueError("Epoch budget must cover every training row")
        self.size, self.budget, self.seed, self.epoch = size, budget, seed, 0

    def set_epoch(self, epoch):
        self.epoch = epoch

    def __len__(self):
        return self.budget

    def __iter__(self):
        indices = list(range(self.size))
        padding = torch.Generator().manual_seed(self.seed + 100000 + self.epoch)
        while len(indices) < self.budget:
            take = min(self.size, self.budget - len(indices))
            indices.extend(torch.randperm(self.size, generator=padding)[:take].tolist())
        order = torch.randperm(self.budget, generator=torch.Generator().manual_seed(self.seed + self.epoch))
        return iter(indices[i] for i in order.tolist())
