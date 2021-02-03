#!/usr/bin/env python
# coding: utf-8

# # Packages

# In[1]:


import numpy as np
import tifffile as tiff
import os
from tqdm import tqdm
import pandas as pd
import pickle
import time
import cv2

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms


# # Utility Functions

# In[132]:


def mask_rle_to_2d(rle_mask, dx, dy):
    """
    converts mask from run length encoding to 2D numpy array
    """

    mask = np.zeros(dx*dy, dtype=np.uint8)
    s = rle_mask.split()  # split the rle encoding

    for i in range(len(s)//2):
        start = int(s[2*i])-1
        length = int(s[2*i+1])
        mask[start:start+length] = 1

    mask = mask.reshape(dy, dx).T

    return mask
        
    
def mask_2d_to_rle(mask_2d):
    """
    Takes a 2D mask of 0/1 and returns the run length encoded form

    """
    mask = mask_2d.T.reshape(-1) #order by columns and flatten to 1D
    mask_padded = np.pad(mask, 1) #pad zero on both sides
    #find the start positions of the 1's
    starts = np.where(mask_padded[:-1] == 0 & (mask_padded[1:] == 1))[0]
    #find the end positions of 1's for each run
    ends = np.where((mask_padded[:-1] == 1) & (mask_padded[1:] == 0))[0]
    
    rle = np.zeros(2*len(starts))
    print(starts.shape, ends.shape, rle.shape)
    rle[::2] = starts
    #length of each run = end position - start position
    rle[1::2] = ends - starts

    rle = ' '.join(str(x) for x in rle)

    return rle

def rle_encode(mask_2d):
    """
        Takes a 2D mask of 0/1 and returns the run length encoded form

    """
    mask = mask_2d.T.reshape( -1 )  # order by columns and flatten to 1D
    mask_padded = np.pad( mask, 1 )  # pad zero on both sides
    runs = np.where(mask_padded[1:] != mask_padded[:-1])[0] + 1
    runs[1::2] -= runs[::2]
    rle = ' '.join( str( x ) for x in runs )
    return rle

# In[3]:


def get_padsize(img, reduce, sz):

    shape = img.shape
    print(shape)

    pad0 = (reduce*sz - shape[0] % (reduce*sz)) % (reduce*sz)
    pad1 = (reduce*sz - shape[1] % (reduce*sz)) % (reduce*sz)
    pad_x = (pad0//2, pad0-pad0//2)
    pad_y = (pad1//2, pad1-pad1//2)

    return pad_x, pad_y


def check_threshold(img_BGR, sat_threshold, pixcount_th):

    """
    checks if an input image passes the threshold conditions:
    conditions:
    not black--> sum of pixels exceeed a threshold = pixcount_th
    saturation --> number of pixels with saturation > sat_threshold exceeds pixcount_th
    Returns:
    True if both conditions are met else False
    """
    #if most of the pixels are black, return False
    #edge of each image is typically black
    if img_BGR.sum() < pixcount_th:
        return False

    #convert to hue, saturation, Value in openCV
    hsv = cv2.cvtColor(img_BGR, cv2.COLOR_BGR2HSV)
    h, s, v = cv2.split(hsv)
    # if less than prefined number of values are above a saturation threshold, return False
    #this is typically the gray background around the biological object
    if (s > sat_threshold).sum() < pixcount_th:
        return False

    return True


# In[111]:


class Image():
    def __init__(self, img, img_name =None, sat_threshold=40, pixcount_th=200):
        self.img = img
        self.shape = img.shape
        self.name = img_name

        self.image_reshape()
        self.dx = self.shape[0]
        self.dy = self.shape[1]

        self.tile_size = None

        self.pad_x = None
        self.pad_y = None
        self.tiled_img = None

        self.mask_rle = None
        self.mask_2d = None
        self.tiled_mask = None
        
    
    def image_reshape(self):
        
        if len(self.shape) == 5:
            self.img = np.transpose(self.img.squeeze(), (1, 2, 0))
            self.shape = self.img.shape
            
    
    def split_image_mask_into_tiles(self, reduce=1, sz=512):
     
        self.tile_size = sz

        self.pad_x, self.pad_y = get_padsize(self.img, reduce, sz)
        print(self.pad_x, self.pad_y)
        #Create padded Image and padded mask2D
        img_padded  = np.pad(self.img, [self.pad_x, self.pad_y, (0, 0)], constant_values=0)
        
        print("shape of image after padding:: ", img_padded.shape,
            img_padded.shape[0]//sz, img_padded.shape[1]//sz)

        #tile the padded image
        img_reshaped = img_padded.reshape(
            img_padded.shape[0]//sz, sz, img_padded.shape[1]//sz, sz, 3)
        img_reshaped = img_reshaped.transpose(0, 2, 1, 3, 4).reshape(-1, sz, sz, 3)
        self.tiled_img = img_reshaped

        '''
        mask_padded = np.pad(self.mask_2d, [self.pad_x, self.pad_y], constant_values = 0)
        print("shape of mask after padding:: ", mask_padded.shape,
              mask_padded.shape[0]//sz, mask_padded.shape[1]//sz)
        #tile the padded mask2D
        mask_reshaped = mask_padded.reshape(
            mask_padded.shape[0]//sz, sz, mask_padded.shape[1]//sz, sz)
        mask_reshaped = mask_reshaped.transpose(
            0, 2, 1, 3).reshape(-1, sz, sz)

        self.tiled_mask = mask_reshaped
        ''' 
            
        
    def reconstruct_original_from_padded_tiled(self, tiled_padded_mask):
        n = tiled_padded_mask.shape[0] #number of tiles after padding
        tile_size = self.tile_size
        (pad_x_l, pad_x_r) = self.pad_x
        (pad_y_l, pad_y_r) = self.pad_y

        dx_padded = self.dx + pad_x_l + pad_x_r
        dy_padded = self.dy + pad_y_l + pad_y_r

        n_x = dx_padded //tile_size
        n_y = dy_padded//tile_size

        assert (n == n_x*n_y), "dimensions don't match"

        mask_untiled = tiled_padded_mask.reshape(n_x, n_y, tile_size, tile_size)
        mask_untiled = mask_untiled.transpose(0,2,1,3)
        mask_padded = mask_untiled.reshape(n_x*tile_size, n_y*tile_size)

        mask_unpadded = mask_padded[pad_x_l: - pad_x_r, pad_y_l: -pad_y_r]

        assert (self.dx == mask_unpadded.shape[0]),             "shape of original image doesn't match with unpadded mask along dim = 0"
        assert (self.dy == mask_unpadded.shape[1]),             "shape of original image doesn't match with unpadded mask along dim = 1"

        return mask_unpadded


# # Model

# In[38]:


class BaseModel(nn.Module):
    
    def __init__(self):
        super(BaseModel, self).__init__()

    def forward(self):
        raise NotImplementedError

    def __str__(self):
        model_params = filter(lambda x: x.requires_grad, self.parameters())

        return super(BaseModel, self).__str__()
    
    
class Conv2x(nn.Module):
    '''
    preserves the the size of the image
    '''
    def __init__(self, in_ch, out_ch, inner_ch=None):
        super(Conv2x, self).__init__()
        self.in_ch = in_ch
        self.out_ch = out_ch
        self.inner_ch = out_ch//2 if inner_ch is None else inner_ch

        self.conv2d_1 = nn.Conv2d(self.in_ch, self.inner_ch,
                                  kernel_size=3, padding=1, bias=False)
        self.conv2d_2 = nn.Conv2d(self.inner_ch, self.out_ch,
                                  kernel_size=3, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(self.inner_ch)
        self.bn2 = nn.BatchNorm2d(self.out_ch)

    def forward(self, x):
        x = self.conv2d_1(x)
        x = self.bn1(x)
        x = F.relu(x)

        x = self.conv2d_2(x)
        x = self.bn2(x)
        x = F.relu(x)

        return x
    
    
class encoder(nn.Module):

    def __init__(self, in_ch, out_ch):
        super(encoder, self).__init__()
        self.conv2x = Conv2x(in_ch, out_ch)
        self.pool = nn.MaxPool2d(kernel_size=2, ceil_mode=True)

    def forward(self, x):
        x = self.conv2x(x)
        x = self.pool(x)
        return x
    
    
class decoder(nn.Module):

    def __init__(self, in_ch, out_ch):
        super(decoder, self).__init__()
        self.transposeconv = nn.ConvTranspose2d(
            in_ch, in_ch//2, kernel_size=2, stride=2)
        self.conv2x = Conv2x(in_ch, out_ch)


    def forward(self, x_down, x_up, interpolate=True):

        x_up = self.transposeconv(x_up)

        #check for matching dims before concatenating

        if (x_up.size(2) != x_up.size(2)) or (x_up.size(3) != x_up.size(3)):
            if interpolate:
                x_up = F.interpolate(x_up, size=(x_down.size(2), x_down.size(3)),
                mode="bilinear", align_corners=True)
        
        #Concat features from down conv channel and current up-conv
        #along channel dim =1
        x_up = torch.cat([x_up, x_down], dim=1) 
        x_up = self.conv2x(x_up)

        return x_up
    


class UNet(BaseModel):

    def __init__(self, in_ch=3, conv_channels=[16, 32, 64, 128, 256]):
        super(UNet, self).__init__()

        self.conv_channels = conv_channels
        self.conv_start = Conv2x(in_ch, conv_channels[0]) #output_size = input_size
        self.down1 = encoder(conv_channels[0], conv_channels[1])   #output_size = input_size/2
        self.down2 = encoder(conv_channels[1], conv_channels[2])   #output_size = input_size/2
        self.down3 = encoder(conv_channels[2], conv_channels[3])   #output_size = input_size/2
        self.down4 = encoder(conv_channels[3], conv_channels[4])   #output_size = input_size/2

        self.conv_middle = Conv2x(conv_channels[4], conv_channels[4]) #output_size = input_size

        self.up4 = decoder(conv_channels[4], conv_channels[3]) #output_size = input_size * 2
        self.up3 = decoder(conv_channels[3], conv_channels[2]) #output_size = input_size * 2
        self.up2 = decoder(conv_channels[2], conv_channels[1]) #output_size = input_size * 2
        self.up1 = decoder(conv_channels[1], conv_channels[0]) #output_size = input_size * 2

        self.final_conv = nn.Conv2d(self.conv_channels[0], 1, kernel_size=1)

        self.init_params()
    
    
    def init_params(self):
        for module in self.modules():
            if isinstance(module, nn.Conv2d) or isinstance(module, nn.Linear):
                nn.init.kaiming_normal_(module.weight)
                if module.bias is not None:
                    module.bias.data.zero_()

            elif isinstance(module, nn.BatchNorm2d):
                module.weight.data.fill_(1)
                module.bias.data.zero_()
                
    
    
    def forward(self, x):
        # size of x = [B, _, nx, ny]
        
        x1 = self.conv_start(x)  # size of x = [B, self.conv_channels[0], nx, ny]
        x2 = self.down1(x1)  # size of x = [B, self.conv_channels[1], nx/2, ny/2]
        x3 = self.down2(x2)  # size of x = [B, self.conv_channels[2], nx/4, ny/4]
        x4 = self.down3(x3)  # size of x = [B, self.conv_channels[3], nx/8, ny/8]
        x5 = self.down4(x4)  # size of x = [B, self.conv_channels[4], nx/16, ny/16]

        x = self.conv_middle(x5)  # size of x = [B, self.conv_channels[4], nx/16, ny/16]

        x = self.up4(x4, x)       # size of x = [B, self.conv_channels[3], nx/8, ny/8]
        x = self.up3(x3, x)       # size of x = [B, self.conv_channels[2], nx/4, ny/4]
        x = self.up2(x2, x)       # size of x = [B, self.conv_channels[1], nx/2, ny/2]
        x = self.up1(x1, x)       # size of x = [B, self.conv_channels[0], nx, ny]

        x = self.final_conv(x)

        return x


# # Metric
def metric_dice_iou(output, target, smooth = 0.005):
        tp = (output * target).sum() #intersection
        fp = (output * (1.0 - target)).sum() #false positives
        fn = ((1.0 - output) * target).sum() #false negatives
        dice = np.mean((2.0 * tp + smooth) / (2 * tp + fp + fn + smooth))
        iou = np.mean((tp + smooth) / (tp + fp + fn + smooth))
        print(tp, fp, fn, output.sum(), target.sum())

        return dice, iou


# ### Dataset for a single image with batch size =1

class Dataset_Image(Dataset):

    def __init__(self, img_dir, img_file, mean, std_dev, tile_size, sat_threshold=40, pixcount_th=200, transform=None):
        super(Dataset_Image, self).__init__()
        
        self.sat_threshold = sat_threshold
        self.pixcount_th = pixcount_th
        self.transform = transform        
        self.normalize = transforms.Normalize(mean, std_dev)
        
        #read image from disk 
        print("reading image from disk")
        img_raw = tiff.imread(os.path.join(img_dir, img_file))
        raw_file_name = img_file.split('.')[0]
    
        self.raw_img = Image(img_raw,img_name=raw_file_name, sat_threshold=sat_threshold, pixcount_th=pixcount_th)
        #call Image.split_image_mask_into_tiles(self, reduce=1, sz=512) to create
        print("tiling image")
        self.raw_img.split_image_mask_into_tiles(reduce=1, sz=tile_size)

        self.tiled_img = self.raw_img.tiled_img
        
        self.len = self.__len__()

    def __len__(self):
        return len(self.tiled_img)

    def __getitem__(self, idx):
        img = self.tiled_img[idx]
        thresold_pass = check_threshold(img, self.sat_threshold, self.pixcount_th)
        img = self.normalize(transforms.ToTensor()(img))
        
        return img, thresold_pass

# # Perform inference

# In[103]:


def inference_single_image(model, img_dir, img_file=None, tile_size = 512,  user=1):
    '''
    if self.val_loader is None:
        print(f"No val loader exists")
        return {}
    '''
    sat_threshold=40
    pixcount_th=200

    mean = [0.68912, 0.47454, 0.6486]
    std_dev = [0.13275, 0.23647, 0.15536]
    
    sat_threshold=40
    pixcount_th=200
    
    inf_dataset = Dataset_Image(img_dir, img_file, mean, std_dev, tile_size, sat_threshold=sat_threshold, pixcount_th=pixcount_th)
    inf_dataloader = DataLoader(inf_dataset, shuffle=False, batch_size=1)

    with torch.no_grad(): 
        tile_count = inf_dataset.len

        total_tiles = 0
        tiles_idx = []
        print(tile_count)
        mask_pred_tiled = np.empty((tile_count,tile_size,tile_size))
        
        #loop over the tiles
        for i, (img_tile, threshold_pass) in enumerate(inf_dataloader):
            
            #img_BGR = tiled_img[i, :, :, :]
            
            if threshold_pass:
                #model_input = torch.unsqueeze(torch.tensor(img_tile),0)
                if (i%500 == 0):
                    print("mask prediction :: {}".format(i))
                img_tile = img_tile.to(device)
                out = torch.squeeze(model(img_tile))
                out = (out > 0).cpu().numpy().astype(np.uint8)
                #print("shape of model output:: ", out.shape)
                        
            else:
                #print("no inference needed for tile :: {}".format(i))
                out = np.zeros((tile_size, tile_size))

            #accumulate the predictions into a big np array
            mask_pred_tiled[i] = out
        #prediction size = [total_tiles, sz, sz]

        
    return mask_pred_tiled, inf_dataset.raw_img


# In[ ]:

def get_device():
    if torch.cuda.is_available():
        dev = 'cuda:0'
    else:
        dev = 'cpu'
    device = torch.device(dev)

    return device


#specify datadir
user = 2
if user == 0:
    masks_train = '/Users/Ethan/Documents/Documents/Documents - Ethan’s MacBook Pro/python/kidney_files/hubmap-kidney-segmentation/train.csv'
    checkpoint_path = '/Users/Ethan/Documents/Documents/Documents - Ethan’s MacBook Pro/python/kidney_files/hubmap-kidney-segmentation/model_checkpoints.pth'
    datadir_train = '/Users/Ethan/Documents/Documents/Documents - Ethan’s MacBook Pro/python/kidney_files/hubmap-kidney-segmentation/train'

elif user == 1:
    masks_train = '/media/bony/Ganga_HDD_3TB/Ganges_Backup/Machine_Learning/HuBMAP_Hacking_Kidney/hubmap-kidney-segmentation/train.csv/'
    checkpoint_path = '/media/bony/Ganga_HDD_3TB/Ganges_Backup/Machine_Learning/HuBMAP_Hacking_Kidney/hubmap-kidney-segmentation/code/saved_checkpoints/UNet_18_01_08_00_30.pth'
    datadir_train = '/media/bony/Ganga_HDD_3TB/Ganges_Backup/Machine_Learning/HuBMAP_Hacking_Kidney/hubmap-kidney-segmentation/train/'

elif user ==2:
    masks_train = 'C:\Scripts\hubmap\\train\\train.csv'
    checkpoint_path = 'C:\Scripts\hubmap\\code\\saved_checkpoints\\UNet_36_02_01_06_46.pth'
    datadir_train = 'C:\Scripts\hubmap\\train'
    datadir_test = 'C:\Scripts\hubmap\\test'

#instantiate model and load saved model parameters
device = get_device()
model = UNet()
# load state dict from saved model dict
print("loading saved model")
model.load_state_dict(torch.load(checkpoint_path, map_location=torch.device('cpu'))['state_dict'])
model.to(device)
model.eval()

#read an image
img_index = 1
image_files = {}
image_files['train'] = [f for f in os.listdir(datadir_train) if "tiff" in f]
image_files['test'] = [f for f in os.listdir(datadir_test) if "tiff" in f]


subm_dict = {}
masks_train = 'C:\Scripts\hubmap\\train.csv'
mask_rle_df = pd.read_csv(masks_train).set_index('id')
print(mask_rle_df.index)
mode = 'train'
datadir = datadir_train
#datadir = datadir_test
raw_image_files = image_files[mode]
print(raw_image_files, datadir)

for img_index in range(len(raw_image_files)):
#for img_index in range(1):

    img_file = raw_image_files[img_index]

    print("\n************  {} *********** \n".format(img_file))
    #call this function for all the prediction images
    t0 = time.time()

    mask_pred, raw_img_class = inference_single_image(model, datadir, img_file, tile_size=512)
    print("time taken for inference :: ",time.time()-t0)

    mask_pred_2d = raw_img_class.reconstruct_original_from_padded_tiled(mask_pred)
    mask_pred_rle = rle_encode(mask_pred_2d)
    print("positive predictions:: ", np.sum(mask_pred_2d))
    raw_file_name = img_file.split('.')[0]
    subm_dict[img_index] ={'id':raw_file_name, 'predicted':mask_pred_rle}



    if mode == 'train':
        print("calculating metrics")

        #mask_rle_df.head()
        img_file_name = img_file.split('.')[0]
        if img_file_name in mask_rle_df.index:
            mask_rle = mask_rle_df.loc[img_file_name, 'encoding']
            dx = raw_img_class.dx
            dy = raw_img_class.dy

            mask_label_2d = mask_rle_to_2d(mask_rle, dx, dy)

            dice, iou = metric_dice_iou(mask_pred_2d, mask_label_2d, smooth = 0.005)
            print("dice :: {:.4f}, iou :: {:.4f}".format(dice, iou))
        else:
            print("mask label not found for image :: ", img_file_name)


submission = pd.DataFrame.from_dict(subm_dict, orient='index')
submission.to_csv("submission3.csv", index=False)


