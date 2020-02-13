import torch
from torch import autograd
from torch.utils.tensorboard import SummaryWriter
from torchvision import transforms, utils
import numpy as np
import argparse
from omegaconf import OmegaConf
from tqdm import tqdm
import sys
import os
from gelsight_tb.utils.infra import str_to_class, deep_map
from gelsight_tb.models.datasets.transforms import ImageTransform
from gelsight_tb.models.modules.vgg_encoder import pretrained_model_normalize
from gelsight_tb.utils.obs_to_np import denormalize_action


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

        self.train_dataset.dataset.transform = transforms.Compose(
            [
                ImageTransform(transforms.ToPILImage()),
                transforms.RandomApply(
                [
                    ImageTransform(transforms.ColorJitter(brightness=0.3, contrast=0.2, saturation=0.2, hue=0.2)),
                    ImageTransform(transforms.RandomRotation(5))
                ]),
                ImageTransform(transforms.ToTensor()),
                ImageTransform(pretrained_model_normalize)
            ])

        self.val_dataset.dataset.transform = transforms.Compose(
            [
                ImageTransform(transforms.ToPILImage()),
                ImageTransform(transforms.ToTensor()),
                ImageTransform(pretrained_model_normalize)
            ]
        )

        if self.conf.dataset.dataloader_workers > 1:
            self.train_dataloader = torch.utils.data.DataLoader(self.train_dataset, batch_size=conf.model.batch_size,
                                                                num_workers=conf.dataset.dataloader_workers)
            self.val_dataloader = torch.utils.data.DataLoader(self.val_dataset, batch_size=conf.model.batch_size,
                                                              num_workers=conf.dataset.dataloader_workers)
        else:
            self.train_dataloader = torch.utils.data.DataLoader(self.train_dataset, batch_size=conf.model.batch_size)
            self.val_dataloader = torch.utils.data.DataLoader(self.val_dataset, batch_size=conf.model.batch_size)

        self.optimizer = torch.optim.Adam(self.model.parameters())
        self.summary_writer = self._make_summary_writer()
        self.global_step = 0
        self.start_epoch = 0

        if resume_dir is not None:
            self.start_epoch = self._load_most_recent_chkpt()

        self.current_epoch = self.start_epoch

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
        weights_path = os.path.join(self.model.exp_path, 'weights')
        checkpoints = os.listdir(weights_path)
        checkpoints.sort(key=lambda f: int(''.join(filter(str.isdigit, f))))
        most_recent_file = os.path.join(weights_path, checkpoints[-1])
        print(f'Loading checkpoint from file {most_recent_file}')
        checkpoint = torch.load(most_recent_file)
        self.model.load_state_dict(checkpoint['state_dict'])
        self.global_step = checkpoint['global_step']
        return checkpoint['epoch']

    def val(self, verbose=False):
        with autograd.no_grad():
            losses = []
            for batch_idx, batch in enumerate(self.val_dataloader):
                inputs = deep_map(lambda x: x.to(self.device), batch)
                output = self.model(inputs)
                loss = self.model.loss(output, inputs['label'])
                if verbose:
                    true_action_batch = denormalize_action(batch['label'], self.conf.dataset.norm)
                    policy_action_batch = denormalize_action(output.cpu().numpy(), self.conf.dataset.norm)
                    for true_action, policy_action in zip(true_action_batch, policy_action_batch):
                        print('-------------------------------------------')
                        print(f'Expert action was {true_action}')
                        print(f'Policy action was {policy_action}')
                        print('-------------------------------------------')
                losses.append(loss * self._batch_size(batch))
            loss = sum(losses) / len(self.val_dataloader.dataset)
            self.summary_writer.add_scalar('val/loss', loss, self.global_step)

    def train(self):
        with tqdm(total=self.conf.num_epochs, desc="epoch: ") as pbar:
            pbar.update(self.current_epoch)
            while self.current_epoch < self.conf.num_epochs:
                if self.current_epoch > self.start_epoch:
                    self.val()
                self.current_epoch += 1
                if self.current_epoch % self.conf.checkpoint_every == 0:
                    self.model.save_checkpoint({
                        'epoch': self.current_epoch,
                        'global_step': self.global_step,
                        'state_dict': self.model.state_dict(),
                        'optimizer': self.optimizer.state_dict(),
                    }, self.current_epoch)
                self.model.dump_params()
                self._train_one_epoch(self.current_epoch)
                pbar.update(1)

    def _train_one_epoch(self, epoch_num):
        self.model.train()
        epoch_len = len(self.train_dataloader)
        print(epoch_len)
        losses = []
        for batch_idx, batch in tqdm(enumerate(self.train_dataloader)):
            inputs = deep_map(lambda x: x.to(self.device), batch)
            self.optimizer.zero_grad()
            output = self.model(inputs)
            loss = self.model.loss(output, inputs['label'])
            loss.backward()
            self.optimizer.step()
            self.global_step = self.global_step + 1
            losses.append(loss * self._batch_size(batch))
            del output, loss
        loss = sum(losses) / len(self.train_dataloader.dataset)
        self.summary_writer.add_scalar('train/loss', loss, self.global_step)

    @staticmethod
    def _batch_size(batch):
        return batch['label'].shape[0]


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='train policy')
    parser.add_argument('config_file', action='store')
    parser.add_argument('--resume_dir', action='store', type=str, dest='resume_dir')
    parser.add_argument('--val', action='store_true', dest='val')
    args = parser.parse_args()
    try:
        conf = OmegaConf.load(args.config_file)
    except:
        print('Failed to load config, exiting now...')
        sys.exit()

    trainer = Trainer(conf, args.resume_dir)
    if args.val:
        trainer.val(verbose=True)
    else:
        trainer.train()
