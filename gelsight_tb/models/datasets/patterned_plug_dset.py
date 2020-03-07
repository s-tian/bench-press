import deepdish as dd
from gelsight_tb.models.datasets.tb_dataset import TBDataset


class PatternPlugDataset(TBDataset):

    def __init__(self, conf, transform=None):
        super(PatternPlugDataset, self).__init__(conf, transform)

    def setup(self):
        self.total_length = len(self.h5_files)

    def __len__(self):
        return self.total_length

    def __getitem__(self, idx):
        file_name = self.h5_files[idx]
        press, pt, final = dd.io.load(file_name, group=[f'/data/i2', '/data/i6', f'/data/i{self.file_lengths[idx]-1}'])
        pt['raw_images']['gelsight_top'] = press['raw_images']['gelsight_top'] 
        pt['images']['gelsight_top'] = press['images']['gelsight_top'] 

        return self._make_data_point(pt, final)

