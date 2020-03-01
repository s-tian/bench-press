import torch
from torch.utils.data import Dataset
from multiprocessing import Pool
import deepdish as dd
import copy
import glob
import bisect
from gelsight_tb.utils.obs_to_np import *


class GuessY(Dataset):

    def __init__(self, conf, transform=None):
        self.conf = conf
        self.folders = conf.folders
        self.transform = transform
        self.h5_files = []
        for folder in self.folders:
            self.h5_files.extend(glob.glob(f'{folder}**/*.h5'))
        print(f'located {len(self.h5_files)} h5 files!')
        self.total_length = len(self.h5_files)
        #self.compute_dataset_statistics(raw=True)

    def __len__(self):
        return self.total_length

    def __getitem__(self, idx):
        file_name =self.h5_files[idx]
        contents = dd.io.load(file_name, group=f'/data/i2')
        return self._make_data_point(contents)

    def _format_data_point(self, images, state):
        data_point = {
            'images': images,
            'state': state,
            'label': state, 
        }
        if self.transform is None:
            return data_point
        transformed = self.transform(data_point)
        return transformed


    def _make_data_point(self, obs_1):
        images = obs_to_images(obs_1)
        state = obs_to_state(obs_1, self.conf.norms.state_norm).astype(np.float32)
        #actions = obs_to_state(final, self.conf.norms.state_norm).astype(np.float32)
        return self._format_data_point(images, state)


    def compute_dataset_statistics(self, raw=True):

        def get_mean_std(data):
            total_data = np.array(data)
            mean = np.mean(total_data, axis=0)
            std = np.std(total_data, axis=0)
            print(f'mean: {mean}')
            print(f'std: {std}')
            return mean, std

        old_statistics = copy.deepcopy(self.conf.norms)
        if raw:
            self.conf.norms.state_norm.mean = [0] * 4
            self.conf.norms.state_norm.scale = [1] * 4
            self.conf.norms.action_norm.mean = [0] * 4
            self.conf.norms.action_norm.scale = [1] * 4
        total_states = []
        total_actions = []
        total_images = []
        for idx in range(len(self)):
            data_point = self[idx]
            total_states.append(data_point['state'])
            total_actions.append(data_point['label'])
            total_images.append(data_point['images'][2])
        print('state statistics')
        get_mean_std(total_states)
        print('action statistics')
        get_mean_std(total_actions)
        print('image 0 stats')
        get_mean_std(total_images)

        # restore previous stats
        self.conf.norm = old_statistics
