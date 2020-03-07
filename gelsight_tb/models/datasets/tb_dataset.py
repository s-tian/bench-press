import torch
from torch.utils.data import Dataset
from multiprocessing import Pool
import deepdish as dd
import glob
import bisect
from gelsight_tb.utils.obs_to_np import *
from gelsight_tb.models.datasets.base_dataset import BaseDataset


class TBDataset(BaseDataset):

    def __init__(self, conf, transform=None):
        super(TBDataset, self).__init__(conf, transform)

    def setup(self):
        self.file_len_cumsum = np.cumsum(np.array(self.file_lengths))
        self.total_length = self.file_len_cumsum[-1]

    def compute_file_and_offset(self, idx):
        file_index = bisect.bisect_right(self.file_len_cumsum, idx)
        file_name = self.h5_files[file_index]
        if file_index == 0:
            sub_index = idx
        else:
            sub_index = idx - self.file_len_cumsum[file_index - 1]
        return file_name, sub_index

    def __getitem__(self, idx):
        file_name, sub_index = self.compute_file_and_offset(idx)

        if self.conf.predict_final_action:
            contents = dd.io.load(file_name, group=[f'/data/i{sub_index}', f'/data/i{sub_index+1}'])
            contents2 = dd.io.load(file_name, group=f'/data/i{self.file_lengths[file_index]-1}')
            data_point = self._make_data_point(contents[0], contents2)
        elif self.conf.use_initial_press:
            contents = dd.io.load(file_name, group=[f'/data/i{sub_index}', f'/data/i{sub_index+1}', f'/data/i2'])
            contents[0]['raw_images']['gelsight_top'] = contents[2]['raw_images']['gelsight_top'] 
            contents[0]['images']['gelsight_top'] = contents[2]['images']['gelsight_top'] 
            contents2 = dd.io.load(file_name, group=f'/data/i{self.file_lengths[file_index]-1}')
            data_point = self._make_data_point(contents[0], contents2)
        else:
            contents = dd.io.load(file_name, group=[f'/data/i{sub_index}', f'/data/i{sub_index+1}'])
            data_point = self._make_data_point(contents[0], contents[1])
        return data_point

    def _make_data_point(self, obs_1, final):
        images = obs_to_images(obs_1)
        state = obs_to_state(obs_1, self.conf.norms.state).astype(np.float32)
        actions = obs_to_action(obs_1, final, self.conf.norms.label).astype(np.float32)
        return self._format_data_point(images, state, actions)

