import torch
import torch.nn as nn
from omegaconf import OmegaConf


class Model(nn.Module):

    def __init__(self, conf):
        super(Model, self).__init__()
        self.conf = conf
        self.exp_path = self.conf.exp_path

    def dump_params(self):
        OmegaConf.save(self.conf, f"{self.exp_path}/conf.yaml")
