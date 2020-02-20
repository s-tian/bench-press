
from gelsight_tb.models.datasets.tb_dataset import TBDataset


class TBDatasetSubset(TBDataset):

    def __init__(self, conf, filter_fn, transform=None):
        super(TBDatasetSubset, self).__init__(conf, transform=transform)
        self.filter_fn = filter_fn
        self.subset_inds = self.get_filter_idxs(self.filter_fn)
        self.subset_len = len(self.subset_inds)

    def get_filter_idxs(self):
        return [i for i in range(len(self)) if self.filter_fn(self[i], self.conf)]

    def __len__(self):
        return self.subset_len

    def __getitem__(self, idx):
        return super(TBDatasetSubset, self).__getitem__(self.subset_inds[idx])


