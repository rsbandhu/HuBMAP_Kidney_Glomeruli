import torch
import time
import numpy as np
from tqdm import tqdm
import json
from torch.utils.data import DataLoader
import torchvision
import os
import logging
import random

#os.environ['KMP_DUPLICATE_LIB_OK']='True'

from base.base_trainer import BaseTrainer
from utils import dataloader_old, loss, metrics, helpers
from models import unet, deeplab_V3


# install the version of pytorch for gpu from this site https://pytorch.org/get-started/previous-versions/
class Trainer(BaseTrainer):

    def __init__(self, model, loss_fn, config, train_loader, val_loader=None):
        super(Trainer, self).__init__(
            model, loss_fn, config, train_loader, val_loader)

        self.optimizer = loss.use_optimizer(model, config)
        self.bce_dice_ratio = config['loss']["bce_dice_ratio"]
        self.loss_alpha = config['loss']['focal_dice']['alpha']
        self.loss_gamma = config['loss']['focal_dice']['gamma']
        self.ratio_focal_dice = config['loss']['focal_dice']['ratio_focal_dice']

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
            #model_out = self.model(img)['out']
            model_out = self.model( img )
            #print((model_out.shape))
            out = torch.squeeze(model_out, 1)
            
            #train_loss = self.loss(out, mask, self.bce_dice_ratio)

            #loss function for mixture of focal loss and dice loss
            train_loss = self.loss(out, mask, self.loss_alpha, self.loss_gamma, self.ratio_focal_dice)

            self.total_loss.update(train_loss.item(), batch_size)

            train_loss.backward() #perform backprop
            self.optimizer.step() #update parameters

            #tbar.set_description( f"Train: Epoch: {epoch}, Avg Loss: {self.total_loss.avg:.5f}" )
            tbar.set_description( "Train: Epoch: {}, Avg Loss: {:.5f}".format( epoch, self.total_loss.avg ) )
            #if (i  > 10):
            #    print(f"epoch: {epoch}, batch : {i}, train loss: {train_loss.item(): .5f}, train average loss: {self.total_loss.avg: .5f}")
            #    break

        logging.info("Train: Epoch: {}, Avg Loss: {:.5f}".format(epoch, self.total_loss.avg))
        return self.total_loss.avg

    def _val_epoch(self, epoch):
        if self.val_loader is None:
            print("No val loader exists")
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

                # loss function for mixture of BCE with Logits loss and dice loss
                #val_loss = self.loss(out, mask, self.bce_dice_ratio)

                # loss function for mixture of focal loss and dice loss
                val_loss = self.loss(out, mask, self.loss_alpha, self.loss_gamma, self.ratio_focal_dice)

                self.total_loss.update(val_loss.item(), batch_size)

                #get the binary masks, ground truth and prediction
                mask_label = mask.cpu().numpy().astype(np.uint8)
                mask_pred = (out >0).cpu().numpy().astype(np.uint8)

                #calculate and update average metrics
                dice, iou = metrics.metric_dice_iou(mask_pred, mask_label)
                self.dice.update(dice, batch_size)
                self.iou.update(iou, batch_size)

                #if (i%10 == 0):
                #    print(f"epoch: {epoch}, batch : {i}, val loss: {val_loss.item()}, val average loss: {self.total_loss.avg}")
                #tbar.set_description(f"Val: Epoch: {epoch}, Avg Loss: {self.total_loss.avg:.5f}")
                tbar.set_description( "Val: Epoch: {}, Avg Loss: {:.5f}, Average Dice: {:.3f}, Average IoU: {:.3f}".format(epoch, self.total_loss.avg, self.dice.avg, self.iou.avg) )

        logging.info("Val: Epoch: {}, Avg Loss: {:.5f}, Average Dice: {:.3f}, Average IoU: {:.3f}".format( epoch,
                                                                self.total_loss.avg, self.dice.avg, self.iou.avg ) )
        return self.total_loss.avg

    def _reset_metrics(self):
        self.total_loss = metrics.AverageMeter()
        self.dice = metrics.AverageMeter()
        self.iou = metrics.AverageMeter()

def main():
    root_dir = 'C:\Scripts\hubmap\code'
    data_dir = 'C:\Scripts\hubmap\\train\\tiled_thresholded_512'
    model_configs = {"unet": 'config_Unet.json', "deeplabV3": 'config_dlv3.json', "deeplabV3_resnet101": "config_dlv3_resnet101.json"}

    model_type = "unet"
    #model_type = "deeplabV3"
    config = json.load(open(model_configs[model_type]))

    model = unet.UNet()
    #model = deeplab_V3.DeepLab( 1, backbone='resnet', freeze_bn=True )
    #model = torchvision.models.segmentation.deeplabv3_resnet50( pretrained=False, progress=True, num_classes=1, aux_loss=None)


    #set logger
    log_folder = config['trainer']['log_dir']
    log_dir = os.path.join( os.getcwd(), log_folder )
    log_file = config['name']+"_"+str(int(time.time()))+".log"
    helpers.set_logger(log_dir, log_file)

    mean = [0.68912, 0.47454, 0.6486]
    std_dev = [0.13275, 0.23647, 0.15536]

    #full dataset with training images and masks
    dataset = dataloader_old.Dataset_Image_mask(data_dir, mean, std_dev)

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
    test_ds.transform = None #do not transform image during validation



    b_size = config["train_loader"]["args"]["batch_size"]
    train_loader = DataLoader(
        train_ds, batch_size=b_size, shuffle=True, num_workers=0)
    b_size = config["val_loader"]["args"]["batch_size"]
    val_loader = DataLoader(
        test_ds, batch_size=b_size, shuffle=True, num_workers=0)

    #checkpoint_file = 'C:\Scripts\hubmap\\code\\saved_checkpoints\\UNet_18_01_08_00_30.pth'

    #loss_fn = loss.loss_fn  #mixture of BCEWithLogits and Dice Loss
    logging.info("Using mixture of focal loss and dice loss")
    loss_fn = loss.loss_focal_dice # mixture of Binary Focal Loss and Dice Loss

    trainer = Trainer(model, loss_fn, config, train_loader, val_loader)
    logging.info("Training on device: {}".format(trainer.device))
    logging.info("model name :: {} ".format(config['name']))
    logging.info("****  Configuration settings\n")
    for item in config:
        logging.info('{}  :: {}'.format(item, config[item]))
    logging.info("\n")

    trainer.train()

    # if trainer._resume_checkpoint(checkpoint_file):
    #     print('resuming from a previous checkpoint')
    #     trainer.train()
    # else:
    #     print("not continuing training")

if __name__ == "__main__":
    main()
