import numpy as np
import torch
from torch.utils.data import DataLoader
from torch.utils.data.sampler import SubsetRandomSampler


class BaseDataLoader(DataLoader):
    def __init__(self, dataset, batch_size, shuffle, num_workers, split_ratio):
        self.dataset = dataset
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.N_tot = len(dataset)
        self.train_samples = len(dataset)
        self.train_sampler, self.val_sampler = self._split_sampler(split_ratio)

        self.init_kawrgs = {
            'dataset':dataset, 'batch_size':batch_size, 'shuffle':self.shuffle,
            'num_workers':num_workers
        }
        super().__init__(sampler=self.train_sampler, **self.init_kawrgs)

    def _split_sampler(self, split_ratio):
        if split_ratio == 0:
            return None, None
        
        idx_full = np.arange(self.N_tot)
        np.random.seed(0)
        np.random.shuffle(idx_full)

        if isinstance(split_ratio, int):
            assert split_ratio > 0
            assert split_ratio < self.N_tot, "Val set size needs to be less than totla number of examples available"
            val_len = split_ratio
        else:
            val_len = int(split_ratio*self.N_tot)

        val_idx = idx_full[:val_len]
        train_idx = np.delete(idx_full, np.arange(0, val_len))

        train_sampler = SubsetRandomSampler(train_idx)
        val_sampler = SubsetRandomSampler(val_idx)

        self.train_samples = self.N_tot - val_len

        return train_sampler, val_sampler

    def val_split(self):
        if self.val_sampler is None:
            return None
        else:
            return DataLoader(sampler=self.val_sampler, **self.init_kawrgs)

