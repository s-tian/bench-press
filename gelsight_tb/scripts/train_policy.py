import torch
import argparse
from omegaconf import OmegaConf
from tqdm import tqdm
import sys
import os
from gelsight_tb.utils.infra import str_to_class, deep_map


class Trainer:

    def __init__(self, conf, resume_dir):
        torch.manual_seed(conf.seed)
        if torch.cuda.is_available():
            self.device = torch.device('cuda')
        else:
            self.device = torch.device('cpu')
        self.model_class = str_to_class(conf.model.type)
        self.model = self.model_class(conf.model).to(self.device)
        self.dataset_class = str_to_class(conf.dataset.type)
        self.dataset = self.dataset_class(conf.dataset)
        self.total_dataset_len = len(self.dataset)
        self.train_val_split = [int(self.conf.train_frac * self.total_dataset_len),
                                self.total_dataset_len - int(self.conf.train_frac * self.total_dataset_len)]
        self.train_dataset, self.val_dataset = torch.utils.data.random_split(self.dataset, self.train_val_split)

        self.optimizer = torch.optim.Adam(self.model.parameters())
        self.global_step = None

    def save_checkpoint(self, d, epoch_num):
        folder_name = os.path.join(self.exp_path, 'weights')
        os.makedirs(folder_name, exist_ok=True)
        torch.save(d, os.path.join(folder_name, f'{epoch_num}.pth'))

    def train(self, start_epoch):
        for epoch in range(start_epoch, self._hp.num_epochs):
            if epoch > start_epoch:
                self.val(not (epoch - start_epoch) % 3)
            self.save_checkpoint({
                'epoch': epoch,
                'global_step': self.global_step,
                'state_dict': self.model.state_dict(),
                'optimizer': self.optimizer.state_dict(),
            })
            self.model.dump_params(self._hp.exp_path)
            self.train_epoch(epoch)

    def _train_one_epoch(self, epoch_num):
        self.model.train()
        epoch_len = len(self.train_dataloader)
        for batch_idx, batch in tqdm(enumerate(self.train_dataloader)):
            inputs = deep_map(lambda x: x.to(self.device), batch)
            self.optimizer.zero_grad()
            output = self.model(inputs)
            loss = self.model.loss(output, inputs['label'])
            loss.backward()
            self.optimizer.step()
            del output, loss
            self.global_step = self.global_step + 1


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
