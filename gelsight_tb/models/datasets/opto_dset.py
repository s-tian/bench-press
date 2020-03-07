from torch.utils.data import Dataset
from multiprocessing import Pool
import deepdish as dd
import copy
import glob
from gelsight_tb.utils.obs_to_np import *
from gelsight_tb.models.datasets.patterned_plug_dset import PatternPlugDataset


class OptoforceDataset(PatternPlugDataset):

    def __init__(self, conf, transform=None):
        self.norm_keys.extend(['opto_1', 'opto_2'])
        super(OptoforceDataset, self).__init__(conf, transform)

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
        press_opto = obs_to_opto(press, self.conf.norms.opto_press_norm)
        curr_opto = obs_to_opto(obs_1, self.conf.norms.opto_curr_norm)
        return self._format_data_point(images, state, actions, press_opto, curr_opto)

