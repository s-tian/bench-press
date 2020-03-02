import torch
from torch.utils.data import Dataset
from multiprocessing import Pool
import deepdish as dd
import copy
import glob
import bisect
from gelsight_tb.utils.obs_to_np import *


class PatternPlugDset(Dataset):

    def __init__(self, conf, transform=None):
        self.conf = conf
        self.folders = conf.folders
        self.transform = transform
        self.h5_files = []
        for folder in self.folders:
            self.h5_files.extend(glob.glob(f'{folder}**/*.h5'))
        self.file_lengths = self._get_file_lengths()
        print(f'located {len(self.h5_files)} h5 files!')
        self.total_length = len(self.h5_files)
        #self.compute_dataset_statistics(raw=True)

    @staticmethod
    def _get_ind_file_len(file):
        contents = dd.io.load(file)
        return len(contents) - 1

    def _get_file_lengths(self):
        with Pool(self.conf.dataloader_workers) as pool:
            file_lengths = pool.map(self._get_ind_file_len, self.h5_files)
        return file_lengths

    def __len__(self):
        return self.total_length

    def __getitem__(self, idx):
        file_name = self.h5_files[idx]
        press, pt, final = dd.io.load(file_name, group=[f'/data/i2', '/data/i6', f'/data/i{self.file_lengths[idx]-1}'])
        pt['raw_images']['gelsight_top'] = press['raw_images']['gelsight_top'] 
        pt['images']['gelsight_top'] = press['images']['gelsight_top'] 

        return self._make_data_point(pt, final, press)

    def _format_data_point(self, images, state, action, opto1, opto2):
        data_point = {
            'images': images,
            'state': state,
            'opto_1': opto1,
            'opto_2': opto2,
            'label': action, 
        }
        if self.transform is None:
            return data_point
        transformed = self.transform(data_point)
        return transformed

    def _make_data_point(self, obs_1, final, press):
        images = obs_to_images(obs_1)
        state = obs_to_state(obs_1, self.conf.norms.state_norm).astype(np.float32)
        actions = obs_to_action(obs_1, final, self.conf.norms.action_norm).astype(np.float32)
        press_opto = obs_to_opto(press, self.conf.norms.opto_norm)
        curr_opto = obs_to_opto(obs_1, self.conf.norms.opto_norm)
        #actions = obs_to_state(final, self.conf.norms.state_norm).astype(np.float32)
        return self._format_data_point(images, state, actions, press_opto, curr_opto)

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
            self.conf.norms.opto_norm.mean = [0] * len(self.conf.norms.opto_norm.mean)
            self.conf.norms.opto_norm.scale = [1] * 4
        total_states = []
        total_actions = []
        total_images = []
        total_opto = []
        for idx in range(len(self)):
            data_point = self[idx]
            total_states.append(data_point['state'])
            total_actions.append(data_point['label'])
            total_images.append(data_point['images'][2])
            if self.conf.optoforce:
                total_opto.append(data_point['opto'])
        print('state statistics')
        get_mean_std(total_states)
        print('action statistics')
        get_mean_std(total_actions)
        print('opto statistics')
        get_mean_std(total_opto)
        print('image 0 stats')
        get_mean_std(total_images)

        # restore previous stats
        self.conf.norms = old_statistics
