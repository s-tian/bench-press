import torch
from torch import autograd
from torch.utils.tensorboard import SummaryWriter
import numpy as np
import argparse
from omegaconf import OmegaConf
from tqdm import tqdm
import sys
import os
from gelsight_tb.utils.infra import str_to_class, deep_map


class Trainer:

    def __init__(self, conf, resume_dir):
        self.conf = conf
        self._set_seeds()
        if torch.cuda.is_available():
            self.device = torch.device('cuda')
        else:
            self.device = torch.device('cpu')
        self.model_class = str_to_class(conf.model.type)
        self.model = self.model_class(conf.model, resume_dir).to(self.device)
        self.dataset_class = str_to_class(conf.dataset.type)
        self.dataset = self.dataset_class(conf.dataset)
        self.total_dataset_len = len(self.dataset)
        self.train_val_split = [int(self.conf.train_frac * self.total_dataset_len),
                                self.total_dataset_len - int(self.conf.train_frac * self.total_dataset_len)]
        self.train_dataset, self.val_dataset = torch.utils.data.random_split(self.dataset, self.train_val_split)
        self.train_dataloader = torch.utils.data.DataLoader(self.train_dataset, batch_size=conf.model.batch_size)
        self.val_dataloader = torch.utils.data.DataLoader(self.val_dataloader, batch_size=conf.model.batch_size)

        self.optimizer = torch.optim.Adam(self.model.parameters())
        self.summary_writer = self._make_summary_writer()
        self.global_step = None

        if resume_dir is not None:
            self._load_most_recent_chkpt()

    def _make_summary_writer(self):
        folder_name = os.path.join(self.model.exp_path, 'logs')
        os.makedirs(folder_name, exist_ok=True)
        return SummaryWriter(folder_name)

    def _set_seeds(self):
        torch.manual_seed(self.conf.seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
        np.random.seed(self.conf.seed)

    def _load_most_recent_chkpt(self):
        checkpoints = os.listdir(os.path.join(self.model.exp_path, 'weights'))
        checkpoints.sort(key=lambda f: int(filter(str.isdigit, f)))
        most_recent_file = checkpoints[-1]
        print(f'Loading checkpoint from file {most_recent_file}')
        checkpoint = torch.load(most_recent_file)
        self.model.load_state_dict(checkpoint['state_dict'])
        self.global_step = checkpoint['global_step']
        return checkpoint['epoch']

    def val(self):
        with autograd.no_grad():
            losses = []
            for batch_idx, batch in tqdm(enumerate(self.val_dataloader)):
                inputs = deep_map(lambda x: x.to(self.device), batch)
                output = self.model(inputs)
                loss = self.model.loss(output, inputs['label'])
                losses.append(loss * batch.shape[0])
            loss = sum(losses) / len(self.val_dataloader)
            self.summary_writer.log_scalar('val/loss', loss, self.global_step)

    def train(self, start_epoch):
        for epoch in range(start_epoch, self._hp.num_epochs):
            if epoch > start_epoch:
                self.val()
            self.model.save_checkpoint({
                'epoch': epoch,
                'global_step': self.global_step,
                'state_dict': self.model.state_dict(),
                'optimizer': self.optimizer.state_dict(),
            })
            self.model.dump_params(self.exp_path)
            self._train_one_epoch(epoch)

    def _train_one_epoch(self, epoch_num):
        self.model.train()
        epoch_len = len(self.train_dataloader)
        losses = []
        for batch_idx, batch in tqdm(enumerate(self.train_dataloader)):
            inputs = deep_map(lambda x: x.to(self.device), batch)
            self.optimizer.zero_grad()
            output = self.model(inputs)
            loss = self.model.loss(output, inputs['label'])
            loss.backward()
            self.optimizer.step()
            del output, loss
            self.global_step = self.global_step + 1
            losses.append(loss * batch.shape[0])
        loss = sum(losses) / len(self.train_dataloader)
        self.summary_writer.log_scalar('train/loss', loss, self.global_step)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='train policy')
    parser.add_argument('config_file', action='store')
    parser.add_argument('--resume_dir', action='store', type=str, dest="resume_dir")
    args = parser.parse_args()
    try:
        conf = OmegaConf.load(args.config_file)
    except:
        print('Failed to load config, exiting now...')
        sys.exit()

    train(conf, args.resume_dir)
