import numpy as np
import pandas as pd 
import os, sys
import tifffile as tiff
import zipfile
import json
import pickle

import preprocessing_utils as utils

#sys.path.append('/home/bony/python-virtual-environments/hubmap/lib/python3.7/site-packages')

import cv2


class Image():
    def __init__(self, img, img_name =None):
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
        '''
        return the shape of an image in the format [x_shape, y_shape, color_channels]
        some images have the shape [1,1,3,x,y]
        change them to have the shape = [x,y,3]
        reshape image accordingly

        returns:
        shape: new shape in the form [x,y,c]
        reshaped image
        '''
        if len(self.shape) == 5:
            self.img = np.transpose(self.img.squeeze(), (1, 2, 0))
            self.shape = self.img.shape

    def split_image_mask_into_tiles(self, reduce=1, sz=512):
        """
        Takes an input image of shape [dx, dy,3]
        pads it on all 4 sides by zeros so that final dx and dy are integral multiple of sz=256
        Then reshapes the image into [-1, sz, sz, 3]
        The first dimennsion is the number of images of size [sz, sz, 3] we get from the original image

        Returns:
        a numpy arr ay of shape [-1, sz, sz, 3]
        """
        self.tile_size = sz

        self.pad_x, self.pad_y = utils.get_padsize(self.img, reduce, sz)
        print(self.pad_x, self.pad_y)
        #Create padded Image and padded mask2D
        img_padded  = np.pad(self.img, [self.pad_x, self.pad_y, (0, 0)], constant_values=0)
        mask_padded = np.pad(self.mask_2d, [self.pad_x, self.pad_y], constant_values = 0)

        print("shape of image after padding:: ", img_padded.shape,
            img_padded.shape[0]//sz, img_padded.shape[1]//sz)

        print("shape of mask after padding:: ", mask_padded.shape,
              mask_padded.shape[0]//sz, mask_padded.shape[1]//sz)

        #tile the padded image
        img_reshaped = img_padded.reshape(
            img_padded.shape[0]//sz, sz, img_padded.shape[1]//sz, sz, 3)
        img_reshaped = img_reshaped.transpose(0, 2, 1, 3, 4).reshape(-1, sz, sz, 3)

        #tile the padded mask2D
        mask_reshaped = mask_padded.reshape(
            mask_padded.shape[0]//sz, sz, mask_padded.shape[1]//sz, sz)
        mask_reshaped = mask_reshaped.transpose(
            0, 2, 1, 3).reshape(-1, sz, sz)

        self.tiled_img = img_reshaped
        self.tiled_mask = mask_reshaped

    def save_thresholded_image(self, tiled_threshold_img_dir, mask_tile_dict, sat_threshold=40, pixcount_th=200):
        n = self.tiled_img.shape[0]

        valid_img_count = 0
        valid_idx = []
        print(f"Original tiled image count = {n}")

        for i in range(n):
            img_BGR = self.tiled_img[i, :, :, :]
            if utils.check_threshold(img_BGR, sat_threshold, pixcount_th):
                valid_img_count += 1
                valid_idx.append(i)
                
                #create an id for the image tile
                img_tile_id = f"{self.name}_{str(self.tile_size)}_{str(valid_img_count)}_{str(i)}"
                img_name = img_tile_id+'.png'  # name of the saved image tile

                mask_for_tile = self.tiled_mask[i, :, :]  # get the mask for the tile
                #convert the mask for the tile to rle
                mask_rle = self.mask_2d_to_rle(mask_for_tile)
                #save the rle mask to a dict, key = name of the corresponding image tile
                mask_tile_dict[img_tile_id] = mask_rle

                #if valid_img_count == 1001:
                cv2.imwrite(os.path.join(tiled_threshold_img_dir, img_name), img_BGR)

        print(f"Image count after thresholding = {valid_img_count}")

    def mask_rle_to_2d(self):
        """
        converts mask from run length encoding to 2D numpy array
        """
        dx = self.dx
        dy = self.dy

        """

        mask = np.zeros(dx*dy, dtype=np.uint8)
        s = self.mask_rle.split()  # split the rle encoding

        for i in range(len(s)//2):
            start = int(s[2*i])-1
            length = int(s[2*i+1])
            mask[start:start+length] = 1

        self.mask_2d = mask.reshape(dy, dx).T
        """
        self.mask_2d = utils.mask_rle_to_2d(self.mask_rle, dx, dy)
        
    def mask_2d_to_rle(self, mask_2d):
        """
        Takes a 2D mask of 0/1 and returns the run length encoded form
        """

        mask = mask_2d.T.reshape(-1)  # order by columns and flatten to 1D
        mask_padded = np.pad(mask, 1)  # pad zero on both sides
        #find the start positions of the 1's
        starts = np.where((mask_padded[:-1] == 0) & (mask_padded[1:] == 1))[0]
        #find the end positions of 1's for each run
        ends = np.where((mask_padded[:-1] == 1) & (mask_padded[1:] == 0))[0]

        rle = np.zeros(2*len(starts))
        
        rle[::2] = starts
        #length of each run = end position - start position
        rle[1::2] = ends - starts
        rle = rle.astype(int)
        return rle

def main():
    #setting parameters for tiling and thresholding raw images
    image_size_reduced = 512
    sat_threshold=40
    pixcount_th=200

    #set the directories for input raw images and output tiled images
    datadir_train = '/media/bony/Ganga_HDD_3TB/Ganges_Backup/Machine_Learning/HuBMAP_Hacking_Kidney/hubmap-kidney-segmentation/train'
    masks_train = '/media/bony/Ganga_HDD_3TB/Ganges_Backup/Machine_Learning/HuBMAP_Hacking_Kidney/hubmap-kidney-segmentation/train.csv'
    tiled_threshold_img_dir = os.path.join(datadir_train, 'tiled_thresholded_'+str(image_size_reduced))
    if not os.path.exists(tiled_threshold_img_dir):
        os.makedirs(tiled_threshold_img_dir)

    #read the mask of the images in rle format
    df_train_masks = pd.read_csv(masks_train).set_index('id')

    #get a list of all the raw images
    os.chdir(datadir_train)
    raw_image_files = [f for f in os.listdir(datadir_train) if "tiff" in f]

    mask_tile_dict = {}

    count = 0
    for f in raw_image_files:
        raw_file_name = f.split('.')[0]
        print(raw_file_name)
        
        img_raw = tiff.imread(os.path.join(datadir_train, f))
        raw_img = Image(img_raw, img_name = raw_file_name)
        #get the mask for the image in rle format
        raw_img.mask_rle = df_train_masks.loc[raw_file_name, 'encoding']

        raw_img.mask_rle_to_2d() #convert mask from rle to 2D
        #print(raw_img.mask_rle[:100])
        #print(f"shape of raw image: {raw_img.shape}, of mask 2D : {raw_img.mask_2d.shape}")
        #print(raw_img.__dir__())

        raw_img.split_image_mask_into_tiles(sz=image_size_reduced)
        print(f"shape of tiled image: {raw_img.tiled_img.shape}, of tiled mask 2D : {raw_img.tiled_mask.shape}")

        raw_img.save_thresholded_image(tiled_threshold_img_dir, mask_tile_dict, sat_threshold=sat_threshold, pixcount_th=pixcount_th)
        count += 1

    #save the tiled mask dict into a file for later use

    tiled_mask_rle_file = open(os.path.join(tiled_threshold_img_dir, 'tiled_mask_rle'), 'wb')
    pickle.dump(mask_tile_dict, tiled_mask_rle_file)
    tiled_mask_rle_file.close()

if __name__ == "__main__":
    main()
