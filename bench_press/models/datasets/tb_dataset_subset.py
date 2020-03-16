from tqdm import tqdm
from bench_press.models.datasets.tb_dataset import TBDataset
from pathlib import Path
import pickle as pkl


class TBDatasetSubset(TBDataset):

    def __init__(self, conf, filter_fn, transform=None):
        self.filter_fn = filter_fn
        super(TBDatasetSubset, self).__init__(conf, transform=transform)

    def setup(self):
        super(TBDatasetSubset, self).setup()
        if self.filter_fn is None:
            self.filter_fn = lambda x, y: True
        else:
            self.filter_fn = self.filter_fn()
        self.subset_inds = self.get_filter_idxs()
        self.subset_len = len(self.subset_inds)
        print(f'Created subset of length {self.subset_len}. This is {1.0 * self.subset_len / self.total_length} of the original.')

    def get_filter_idxs(self):
        print('Loading subset filter indices...')
        if self.conf.save_inds_to:
            if Path(self.conf.save_inds_to).is_file():
                print(f'Loading cached indices from {self.conf.save_inds_to}...')
                with open(self.conf.save_inds_to, 'rb') as f:
                    saved_inds = pkl.load(f)
                return saved_inds

        filter_inds = []
        for i in tqdm(range(self.total_length)):
            if self.filter_fn(super(TBDatasetSubset, self).__getitem__(i), self.conf):
                filter_inds.append(i)
        print(f'Caching indices to {self.conf.save_inds_to}...')
        with open(self.conf.save_inds_to, 'wb') as f:
            pkl.dump(filter_inds, f)
        return filter_inds

    def __len__(self):
        return self.subset_len

    def __getitem__(self, idx):
        return super(TBDatasetSubset, self).__getitem__(self.subset_inds[idx])

