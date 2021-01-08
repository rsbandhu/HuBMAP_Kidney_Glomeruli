import numpy as np
import json
import torch
import os
import logging
import datetime


class BaseTrainer:

    def __init__(self, model, loss, config, train_loader, val_loader):
        self.model = model
        self.loss = loss
        self.config = config
        self.train_loader = train_loader
        self.val_loader = val_loader

        self.start_epoch = 1

        self.device = self.get_device()

        self.model.to(self.device)

        #Training configs
        cfg_train = config['trainer']
        self.epochs = cfg_train['epochs']
        self.save_period = cfg_train['save_period']

        #Checkpoint configs
        cur_dir = os.curdir
        self.checkpoint_dir = os.path.join(cur_dir, cfg_train["save_dir"])
        print(self.checkpoint_dir)
        if not os.path.exists(self.checkpoint_dir):
            os.makedirs(self.checkpoint_dir)

    def train(self):

        best_val_loss = 0.0

        for epoch in range(self.start_epoch, self.epochs+1):
            print(f"*******   Starting Training epoch :: {epoch}  ***************\n")
            self._train_epoch(epoch)

            # save checkpoint
            if epoch % self.config["trainer"]["save_period"] == 0:
                print(f"*******   Starting Validation, epoch :: {epoch}  ***************\n")
                val_loss = self._val_epoch(epoch)
                if np.log(val_loss) < best_val_loss:
                    best_val_loss = np.log(val_loss)
                    self._save_checkpoints(epoch)

            '''
            # perform validation on the val dataset
            if epoch % self.config['trainer']['val_every_x_epochs'] == 0:
                self._val_epoch(epoch)

            # Check if improvement has stalled on val datatset
            #to be implemented

            '''

    def get_device(self):
        if torch.cuda.is_available():
            dev = 'cuda:0'
        else:
            dev = 'cpu'
        device = torch.device(dev)

        return device

    def _save_checkpoints(self, epoch):
        #create a state dict for saving

        model_state = {
            'epoch':epoch,
            'state_dict':self.model.state_dict(),
            'optimizer':self.optimizer.state_dict(),
            'config':self.config
        }
        savetime = datetime.datetime.now().strftime('%m_%d_%H_%M')
        filename = f"{self.config['name']}_{epoch}_{savetime}"
        filename = os.path.join(self.checkpoint_dir, f'{filename}.pth')
        torch.save(model_state, filename)

    def _resume_checkpoint(self):
        pass

    def _train_epoch(self):
        #implement this in Trainer (sub class of BaseTrainer) 
        raise NotImplementedError

    def _val_epoch(self, epoch):
        raise NotImplementedError
