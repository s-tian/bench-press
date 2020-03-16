import torch
from torch.utils.data import Dataset
from multiprocessing import Pool
import deepdish as dd
import copy
import glob
import bisect
from bench_press.utils.obs_to_np import *


class StateEstimateFromTrajDataset(Dataset):

    def __init__(self, conf, transform=None):
        super(StateEstimateFromTrajDataset, self).__init__(conf, transform)

    def __len__(self):
        return self.total_length

    def __getitem__(self, idx):
        file_name = self.h5_files[idx]
        contents = dd.io.load(file_name, group=f'/data/i2')
        return self._make_data_point(contents)

    def _make_data_point(self, obs_1):
        images = obs_to_images(obs_1)
        state = obs_to_state(obs_1, self.conf.norms.state_norm).astype(np.float32)
        return self._format_data_point(images, state, state)
