import torch
from torch.utils.data import Dataset
import deepdish as dd
import glob
import bisect
from gelsight_tb.utils.obs_to_np import *


class TBDataset(Dataset):

    def __init__(self, conf, transform=None):
        self.conf = conf
        self.folder = conf.folder
        self.transform = transform
        self.h5_files = [folder for folder in glob.glob(f'{self.folder}**/*.h5')]
        print(f'located {len(self.h5_files)} h5 files!')
        self.file_lengths = []
        for f in self.h5_files:
            contents = dd.io.load(f)
            self.file_lengths.append(len(contents)-1)
        self.file_len_cumsum = np.cumsum(np.array(self.file_lengths))
        self.total_length = self.file_len_cumsum[-1]

    def __len__(self):
        return self.total_length

    def __getitem__(self, idx):
        file_index = bisect.bisect(self.file_len_cumsum, idx)
        file_name = self.h5_files[file_index]

        contents = dd.io.load(file_name)
        sub_index = idx - self.file_len_cumsum[file_index]
        data_point = self._make_data_point(contents[sub_index], contents[sub_index+1])
        return data_point

    def _make_data_point(self, obs_1, obs_2):
        images = obs_to_images(obs_1, self.conf.norm)
        state = obs_to_state(obs_1, self.conf.norm).astype(np.float32)
        actions = obs_to_action(obs_1, obs_2, self.conf.norm).astype(np.float32)
        data_point = {
            'images': images,
            'state': state,
            'label': actions
        }
        if self.transform is None:
            return data_point
        transformed = self.transform(data_point)
        return transformed
