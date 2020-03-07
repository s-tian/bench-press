from torch.utils.data import Dataset
from multiprocessing import Pool
import deepdish as dd
import glob
import bisect
import copy
from gelsight_tb.utils.obs_to_np import *
from gelsight_tb.models.datasets.tb_dataset import TBDataset


class StateEstimationDataset(TBDataset):

    """
    Perform state estimation from data collected using random pressing policy
    """

    def __init__(self, conf, transform=None):
        super(StateEstimationDataset, self).__init__(conf, transform)

    @staticmethod
    def _get_ind_file_len(f):
        contents = dd.io.load(f)
        return (len(contents) - 2) // 2

    def __getitem__(self, idx):
        file_name, sub_index = self.compute_file_and_offset(idx)
        contents = dd.io.load(file_name, group=[f'/data/i{2*sub_index + 2}'])
        data_point = self._make_data_point(contents[0])
        return data_point

    def _make_data_point(self, obs_1):
        images = obs_to_images(obs_1)
        state = obs_to_state(obs_1, self.conf.norms.state_norm).astype(np.float32)
        return self._format_data_point(images, state, state)
