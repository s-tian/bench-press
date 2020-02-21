from tqdm import tqdm
from gelsight_tb.models.datasets.tb_dataset import TBDataset


class TBDatasetSubset(TBDataset):

    def __init__(self, conf, filter_fn, transform=None):
        super(TBDatasetSubset, self).__init__(conf, transform=transform)
        self.filter_fn = filter_fn()
        self.subset_inds = self.get_filter_idxs()
        self.subset_len = len(self.subset_inds)
        print(f'Created subset of length {self.subset_len}')
        #self.compute_dataset_statistics(raw=True)

    def get_filter_idxs(self):
        print('Loading subset filter indices')
        filter_inds = []
        for i in tqdm(range(self.total_length)):
            if self.filter_fn(super(TBDatasetSubset, self).__getitem__(i), self.conf):
                filter_inds.append(i)
        return filter_inds

    def __len__(self):
        return self.subset_len

    def __getitem__(self, idx):
        return super(TBDatasetSubset, self).__getitem__(self.subset_inds[idx])

