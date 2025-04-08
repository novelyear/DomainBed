# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved

import torch
from torch.utils.data import DataLoader, RandomSampler, DistributedSampler, BatchSampler

class _InfiniteSampler(torch.utils.data.Sampler):
    """Wraps another Sampler to yield an infinite stream."""
    def __init__(self, sampler):
        self.sampler = sampler

    def __iter__(self):
        while True:
            for batch in self.sampler:
                yield batch

class InfiniteDataLoader:
    def __init__(self, dataset, weights, batch_size, num_workers, sampler=None):
        super().__init__()

        if weights is not None:
            data_sampler = torch.utils.data.WeightedRandomSampler(
                weights, replacement=True, num_samples=batch_size
            )
        elif sampler is not None:
            data_sampler = sampler
        else:
            data_sampler = torch.utils.data.RandomSampler(dataset, replacement=True)

        batch_sampler = torch.utils.data.BatchSampler(
            data_sampler,
            batch_size=batch_size,
            drop_last=True
        )

        self._infinite_iterator = iter(torch.utils.data.DataLoader(
            dataset,
            num_workers=num_workers,
            batch_sampler=_InfiniteSampler(batch_sampler)
        ))

    def __iter__(self):
        while True:
            try:
                yield next(self._infinite_iterator)
            except StopIteration:
                pass

    def __len__(self):
        raise ValueError


class InfiniteDataLoaderWithoutReplacement:
    def __init__(self, dataset, weights, batch_size, num_workers):
        super().__init__()

        if weights is not None:
            sampler = torch.utils.data.WeightedRandomSampler(weights,
                replacement=False,
                num_samples=batch_size)
        else:
            sampler = torch.utils.data.RandomSampler(dataset,
                replacement=False)

        if weights == None:
            weights = torch.ones(len(dataset))

        batch_sampler = torch.utils.data.BatchSampler(
            sampler,
            batch_size=batch_size,
            drop_last=True)

        self._infinite_iterator = iter(torch.utils.data.DataLoader(
            dataset,
            num_workers=num_workers,
            batch_sampler=_InfiniteSampler(batch_sampler)
        ))

    def __iter__(self):
        while True:
            yield next(self._infinite_iterator)

    def __len__(self):
        raise ValueError



class FastDataLoader:
    """
    DataLoader wrapper with slightly improved speed by not respawning worker
    processes at every epoch. Supports DDP by using DistributedSampler if necessary.
    """
    def __init__(self, dataset, batch_size, num_workers, device, is_ddp=False):
        super().__init__()
        self.device = device
        self.is_ddp = is_ddp

        if self.is_ddp:
            self.sampler = DistributedSampler(dataset, shuffle=False)
        else:
            self.sampler = RandomSampler(dataset, replacement=False)

        self.batch_sampler = BatchSampler(
            self.sampler,
            batch_size=batch_size,
            drop_last=False
        )

        self.loader = DataLoader(
            dataset,
            batch_sampler=self.batch_sampler,
            num_workers=num_workers,
            pin_memory=True,
            persistent_workers=True
        )

        self._length = len(self.batch_sampler)

    def __iter__(self):
        for batch in self.loader:
            x, y = batch
            yield x.to(self.device, non_blocking=True), y.to(self.device, non_blocking=True)

    def __len__(self):
        return self._length
