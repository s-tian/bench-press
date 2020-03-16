from torch.utils.data import Dataset
import glob
import deepdish as dd
import numpy as np
from multiprocessing import Pool


class BaseDataset(Dataset):

    """
    Base HDF5 dataset
    """

    def __init__(self, conf, transform=None):
        self.conf = conf
        self.folders = conf.folders
        self.transform = transform
        self.h5_files = []
        self.norm_keys = ['state', 'label']
        for folder in self.folders:
            self.h5_files.extend(glob.glob(f'{folder}**/*.h5'))
        self.file_lengths = self._get_file_lengths()
        self.total_length = len(self.h5_files)
        print(f'located {len(self.h5_files)} h5 files!')
        self.setup()
        self.conf.norms = self.compute_dataset_statistics()

    def setup(self):
        pass

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

    def _get_key_data(self, idx):
        contents = self[idx]
        return {k: contents[k] for k in self.norm_keys}

    def _format_data_point(self, images, state, actions):
        data_point = {
            'images': images,
            'state': state,
            'label': actions
        }
        if self.transform is None:
            return data_point
        transformed = self.transform(data_point)
        return transformed

    def compute_dataset_statistics(self):

        if not self.conf.norms:
            self.conf.norms = dict()

        for key in self.norm_keys:
            if key not in self.conf.norms:
                self.conf.norms[key] = dict()
            init_shape = self[0][key].shape[0]
            self.conf.norms[key].mean = [0] * init_shape
            self.conf.norms[key].std = [1] * init_shape

        with Pool(self.conf.dataloader_workers) as pool:
            data = pool.map(self._get_key_data, range(self.total_length))

        norms = dict()
        for key in self.norm_keys:
            key_data = np.array([x[key] for x in data])
            mean, std = np.mean(key_data, axis=0), np.std(key_data, axis=0)
            norms[key] = dict()
            norms[key]['mean'], norms[key]['std'] = mean.tolist(), std.tolist()
        print(norms)
        return norms
