import numpy as np
from torch.utils.data import DataLoader
from torch.utils.data.dataloader import default_collate
from torch.utils.data.sampler import SubsetRandomSampler, SequentialSampler
from sklearn.model_selection import KFold, train_test_split


class BaseDataLoader(DataLoader):
    """
    Base class for all data loaders
    """
    def __init__(self, dataset, batch_size, shuffle, validation_split, random_seed, num_workers, collate_fn=default_collate):
        self.validation_split = validation_split
        self.shuffle = shuffle

        self.batch_idx = 0
        self.n_samples = len(dataset)
        self.random_seed = random_seed

        # self.sampler, self.valid_sampler, self.test_sampler = self._split_sampler(self.validation_split)

        self.init_kwargs = {
            # 'dataset': dataset,
            'batch_size': batch_size,
            'shuffle': self.shuffle,
            'collate_fn': None,
            'num_workers': num_workers,
        }
        super().__init__(dataset, **self.init_kwargs)

    def _split_sampler(self, split):
        if split == 0.0:
            return None, None


        idx_full = np.arange(self.n_samples)
        np.random.seed(self.random_seed)
        np.random.shuffle(idx_full)

        if isinstance(split, int):
            assert split > 0
            assert split < self.n_samples, "validation set size is configured to be larger than entire dataset."
            len_valid = split
        else:
            len_valid = len_test = int(self.n_samples * split * 0.5)

        valid_idx = idx_full[0:len_valid]
        test_idx = idx_full[len_valid: len_valid+len_test]
        train_idx = np.delete(idx_full, np.arange(0, len_valid*2))

        train_sampler = SequentialSampler(train_idx)
        valid_sampler = SequentialSampler(valid_idx)
        test_sampler = SequentialSampler(test_idx)


        # turn off shuffle option which is mutually exclusive with sampler
        self.shuffle = False
        self.n_samples = len(train_idx)

        return train_sampler, valid_sampler, test_sampler

    def split_validation(self):
        self.sampler, self.valid_sampler, self.test_sampler = self._split_sampler(self.validation_split)
        if self.valid_sampler is None:
            return None
        else:
            return DataLoader(sampler=self.valid_sampler, **self.init_kwargs)

    def split_test(self):
        self.sampler, self.valid_sampler, self.test_sampler = self._split_sampler(self.validation_split)
        if self.test_sampler is None:
            return None
        else:
            return DataLoader(sampler=self.test_sampler, **self.init_kwargs)