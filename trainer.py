import torch
import time
import numpy as np
from tqdm import tqdm
import json
from torch.utils.data import DataLoader
import torchvision

from base.base_trainer import BaseTrainer
from utils import dataloader, loss, metrics
from models import unet

class Trainer(BaseTrainer):

    def __init__(self, model, loss_fn, config, train_loader, val_loader=None):
        super(Trainer, self).__init__(
            model, loss_fn, config, train_loader, val_loader)

        self.optimizer = loss.use_optimizer(model, config)
        self.bce_dice_ratio = config['loss']["bce_dice_ratio"]

    def _train_epoch(self, epoch):
        '''
        train the model for one epoch
        '''
        self.model.train()
        self._reset_metrics()
        tbar = tqdm(self.train_loader, ncols = 100, miniters=50)

        for i, sample_batch in enumerate(tbar):
            self.optimizer.zero_grad()

            img = sample_batch['image']
            mask = sample_batch['mask'].float()

            batch_size = img.shape[0]

            img = img.to(self.device)
            mask = mask.to(self.device)
            #img = img.transpose((0,3,1,2))
            model_out = self.model(img)['out'] #this line is for deeplabV3 from torchvision
            out = torch.squeeze(model_out, 1)
            #out = torch.squeeze(self.model(img), 1)
            
            train_loss = self.loss(out, mask, self.bce_dice_ratio)
            self.total_loss.update(train_loss.item(), batch_size)

            train_loss.backward() #perform backprop
            self.optimizer.step() #update parameters

            tbar.set_description( f"Train: Epoch: {epoch}, Avg Loss: {self.total_loss.avg:.5f}" )
            #if (i % 50 == 0):
            #    print(f"epoch: {epoch}, batch : {i}, train loss: {train_loss.item(): .5f}, train average loss: {self.total_loss.avg: .5f}")
            #    break
        return self.total_loss.avg

    def _val_epoch(self, epoch):
        if self.val_loader is None:
            print(f"No val loader exists")
            return {}

        self.model.eval()
        self._reset_metrics()
        tbar = tqdm(self.val_loader, ncols=100)
        with torch.no_grad():
            for i, sample_batch in enumerate(tbar):
                img = sample_batch['image']
                mask = sample_batch['mask'].float()

                batch_size = img.shape[0]
                img = img.to(self.device)
                mask = mask.to(self.device)

                out = torch.squeeze(self.model(img), 1)
                val_loss = self.loss(out, mask, self.bce_dice_ratio)
                self.total_loss.update(val_loss.item(), batch_size)
                #if (i%10 == 0):
                #    print(f"epoch: {epoch}, batch : {i}, val loss: {val_loss.item()}, val average loss: {self.total_loss.avg}")
                tbar.set_description(f"Val: Epoch: {epoch}, Avg Loss: {self.total_loss.avg:.5f}")

        return self.total_loss.avg

    def _reset_metrics(self):

        self.total_loss = metrics.AverageMeter()

def main():
    root_dir = 'C:\Scripts\hubmap\code'

    data_dir = 'C:\Scripts\hubmap\\train\\tiled_thresholded_512'
    data_dir = '/media/bony/Ganga_HDD_3TB/Ganges_Backup/Machine_Learning/HuBMAP_Hacking_Kidney/hubmap-kidney-segmentation/train/tiled_thresholded_512'

    mean = [0.68912, 0.47454, 0.6486]
    std_dev = [0.13275, 0.23647, 0.15536]

    #full dataset with training images and masks
    dataset = dataloader.Dataset_Image_mask(data_dir, mean, std_dev)

    n_tot = dataset.len

    #SplitS full dataset into train set and test set
    train_test_split = 0.8
    train_count = int(train_test_split * n_tot)

    test_count = dataset.len - train_count

    train_idx = list(np.random.choice(
        range(n_tot), train_count, replace=False))
    test_idx = list(set(range(n_tot)) - set(train_idx))

    print(len(train_idx), len(test_idx), n_tot - len(train_idx) - len(test_idx))

    train_ds = torch.utils.data.Subset(dataset, train_idx)
    test_ds = torch.utils.data.Subset(dataset, test_idx)


    #model = unet.UNet()
    model = torchvision.models.segmentation.deeplabv3_resnet50(
        pretrained=False, progress=True, num_classes=1, aux_loss=None)

    config = json.load(open('config.json'))
    b_size = config["train_loader"]["args"]["batch_size"]
    train_loader = DataLoader(
        train_ds, batch_size=b_size, shuffle=True, num_workers=0)
    b_size = config["val_loader"]["args"]["batch_size"]
    val_loader = DataLoader(
        test_ds, batch_size=b_size, shuffle=True, num_workers=0)

    trainer = Trainer(model, loss.loss_fn, config, train_loader, val_loader)
    print(f"Trainining on device: {trainer.device}")

    trainer.train()

if __name__ == "__main__":
    main()
