import numpy as np
import pandas as pd 
import os, sys
import tifffile as tiff
import zipfile
import json

import torchvision.transforms as transforms

#sys.path.append('/home/bony/python-virtual-environments/hubmap/lib/python3.7/site-packages')

import cv2


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
    starts = np.where(mask_padded[:-1] == 1 & (mask_padded[1:] == 0))[0]
    #find the end positions of 1's for each run
    ends = np.where((mask_padded[:-1] == 0) & (mask_padded[1:] == 1))[0]
    
    rle = np.zeros(2*len(starts))
    print(starts.shape, ends.shape, rle.shape)
    rle[::2] = starts
    #length of each run = end position - start position
    rle[1::2] = ends - starts

    return rle


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

def image_reshape(img):
    '''
    return the shape of an image in the format [x_shape, y_shape, color_channels]
    some images have the shape [1,1,3,x,y]
    change them to have the shape = [x,y,3]
    reshape image accordingly

    returns:
    shape: new shape in the form [x,y,c]
    reshaped image
    '''
    shape = img.shape

    
    if len(img.shape) == 5:
        img = np.transpose(img.squeeze(), (1, 2, 0))
        shape = img.shape

    return shape, img

def split_image_into_tiles(img, mask, reduce=4, sz=256):
    """
    Takes an input image of shape [dx, dy,3]
    pads it on all 4 sides by zeros so that final dx and dy are integral multiple of sz=256
    Then reshapes the image into [-1, sz, sz, 3]
    The first dimennsion is the number of images of size [sz, sz, 3] we get from the original image

    Returns:
    a numpy arr ay of shape [-1, sz, sz, 3]
    """

    shape, img = image_reshape(img)
    
    pad0 = (reduce*sz - shape[0] % (reduce*sz)) % (reduce*sz)
    pad1 = (reduce*sz - shape[1] % (reduce*sz)) % (reduce*sz)
    pad_x = (pad0//2, pad0-pad0//2)
    pad_y = (pad1//2, pad1-pad1//2)
    img_padded = np.pad(img, [pad_x, pad_y, (0, 0)], constant_values=0)
    print("shape of image after padding:: ",img_padded.shape, img_padded.shape[0]//sz, img_padded.shape[1]//sz)

    mask_padded = np.pad(mask, [pad_x, pad_y], constant_values=0) #pad the 2D mask for the image
    print("shape of mask padded ", mask_padded.shape)
    #tile the padded image
    img_reshaped = img_padded.reshape(
        img_padded.shape[0]//sz, sz, img_padded.shape[1]//sz, sz, 3)
    img_reshaped = img_reshaped.transpose(0, 2, 1, 3, 4).reshape(-1, sz, sz, 3)

    #tile the padded mask
    mask_tiled = mask_padded.reshape(
        img_padded.shape[0]//sz, sz, img_padded.shape[1]//sz, sz)
    mask_tiled = mask_tiled.transpose(0,2,1,3).reshape(-1, sz, sz) 

    print(f"shape final tile: {img_reshaped.shape}, shape final mask: {mask_tiled.shape}, number of tiles and mask = {img_reshaped.shape[0]}, {mask_tiled.shape[0]}")

    return img_reshaped, mask_tiled


def save_thresholded_image(img, mask, raw_img_name, sz, output_dir, mask_tile_dict, sat_threshold=40, pixcount_th=200):
    n = img.shape[0]
    valid_img_count = 0
    sat_threshold = 40
    pixcount_th = 200
    valid_idx = []
    print(f"Original tiled image count = {n}")

    for i in range(n):
        img_BGR = img[i, :, :, :]
        if check_threshold(img_BGR, sat_threshold, pixcount_th):
            valid_img_count += 1
            valid_idx.append(i)
            #img_BGR = cv2.imencode('.png', img_BGR)[1]
            #img_out.writestr(f'test_512_{i}.png', img_BGR)
            #create an id for the image tile
            img_tile_id = raw_img_name+'_'+str(sz)+'_'+str(valid_img_count)
            img_name = img_tile_id+'.png' #name of the saved image tile

            mask_for_tile = mask[i, :, :] #get the mask for the tile
            #convert the mask for the tile to rle
            mask_rle = mask_2d_to_rle(mask_for_tile)
            #save the rle mask to a dict, key = name of the corresponding image tile
            mask_tile_dict[img_tile_id]=mask_rle

            #if valid_img_count == 1001:
            cv2.imwrite(os.path.join(output_dir, img_name), img_BGR)
    print(f"Image count after thresholding = {valid_img_count}")

def image_transform(mode='train'):
    transform = None

    v_flip = transforms.RandomVerticalFlip()
    h_flip = transforms.RandomHorizontalFlip()

    if mode == 'train':
        transform = transforms.Compose([v_flip, h_flip])
    
    return transform




def main():

    datadir_train = '/media/bony/Ganga_HDD_3TB/Ganges_Backup/Machine_Learning/HuBMAP_Hacking_Kidney/hubmap-kidney-segmentation/train'
    os.chdir(datadir_train)
    masks_train = '/media/bony/Ganga_HDD_3TB/Ganges_Backup/Machine_Learning/HuBMAP_Hacking_Kidney/hubmap-kidney-segmentation/train.csv'
    image_size = 512
    output_dir = os.path.join(datadir_train, 'tiled_'+str(image_size))
    #os.mkdir(output_dir)
    print(output_dir)
    raw_image_files = [f for f in os.listdir(datadir_train) if "tiff" in f]

    df_train_masks = pd.read_csv(masks_train).set_index('id')
    img_train_list = list(df_train_masks.index)
    print(img_train_list)
    #mask_rle = df_train_masks.loc['2f6ecfcdf', 'encoding']
    mask_tile_dict = {}
    
    #loop over the original image files
    #split each image file into multiple files
    #discard the ones that have too many black pixels or uniform saturation
    for f in raw_image_files:
        raw_file_name = f.split('.')[0]
        print(raw_file_name)

        mask_rle = df_train_masks.loc[raw_file_name, 'encoding']
        #print(raw_image_files)
    
        img_raw = tiff.imread(os.path.join(datadir_train, f))
        [dx, dy, c], img_raw = image_reshape(img_raw)

        #create the 2d mask from rle 
        mask_2d = mask_rle_to_2d(mask_rle, dx, dy)
        print("shape of unpadded 2D mask ", mask_2d.shape)
        #create an array of one more diemension for the different tiles
        tiled_img , tiled_mask = split_image_into_tiles(img_raw, mask_2d, reduce=1, sz=image_size)

        print(tiled_img.shape)
        #save only those tiles that meet the saturation and black pixel count criteria
        #save_thresholded_image(tiled_img, tiled_mask, raw_file_name, image_size, output_dir, mask_tile_dict)
    #cv2.imwrite('test.png', tiled_img[1000,:,:,:])
    #np.save("tiled_mask_dict.npy", mask_tile_dict)
    #print(raw_image_files[0])
    #img_raw = tiff.imread(os.path.join(datadir_train, raw_image_files[0]))
    #print(img_raw.shape)
    
if __name__ == "__main__":
    main()
    
    
original_img_count = 0
original_idx = []
def reconstruct_images(self,dx=img_raw.shape[0],dy=img_raw.shape[1]):
    for i in range(n):
        img_BGR = img[i, :, :, :]
        if check_threshold(img_BGR, sat_threshold, pixcount_th):
            original_img_count += 1
            original_idx.append(i)
            # remove the padding
            img_depadded = (self.img_padded, [-self.pad_x, -self.pad_y])
            mask_depadded = (self.mask_padded, [-self.pad_x, -self.pad_y])
            img_depadded = np.squeeze(img_depadded.shape[0])
            mask_depadded = np.squeeze(mask_depadded.shape[0])
            # reshape
            img_reconstructed_dxdy = img_depadded.shape(img_depadded.shape[0] * 512, img_depadded.shape[1] * 512)
            img_reconstructed = img_reconstructed_dxdy.reshape(dx,dy,3)
            mask_reconstructed_dxdy = mask_depadded.shape(mask_depadded.shape[0] * 512, mask_depadded.shape[1] * 512)
            mask_reconstructed = mask_reconstructed_dxdy.reshape(dx, dy, 3)
            
            print("shape of image after reconstruction:: ", img_reconstructed.shape)
            print("shape of mask after reconstruction:: ", mask_reconstructed.shape)
    
    return img_reconstructed

# convert 2D array to rle
def reconstruct_to_rle():
    starts = np.where((mask_reconstructed[:-1] == 0) & (mask_reconstructed[1:] == 1))[0]
    ends = np.where((mask_reconstructed[:-1] == 1) & (mask_reconstructed[1:] == 0))[0]
    rle = np.zeros(2 * len(starts))
    rle[::2] = starts
    rle[1::2] = ends - starts
    rle = rle.astype(int)
    return rle

