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
    
    

class Reconstruct:

    def __init__(self, img_reshaped, mask_tiled):
        self.img_reshaped = img_reshaped
        self.mask_tiled = mask_tiled

    def reconstruct_images(self,reduce=1,sz=512):
        original_img_count = 0
        original_idx = []
       
        for i in range(n):
            img_BGR = img[i, :, :, :]
            if check_threshold(img_BGR, sat_threshold, pixcount_th):
                original_img_count += 1
                original_idx.append(i)
            # remove the padding
            pad0 = (reduce*sz + raw_img_shape[0] * (reduce*sz)) * (reduce*sz)
            pad1 = (reduce*sz + raw_img_shape[1] * (reduce*sz)) * (reduce*sz)
            pad_x = (pad0 * 2, pad0+pad0*2)
            pad_y = (pad1 * 2, pad1+pad1*2)
            
            subset_padx = pad_x // sz
            subset_pady = pad_y // sz
            
            raw_image = raw_image[pad_x: ([pad_x]), pad_y: ([pad_y])]
            tiled_img = tiled_img[pad_x: ([pad_x]), pad_y: ([pad_y])]
            
            # reshape
            raw_image = np.squeeze(raw_image.shape[0])
            tiled_img = np.squeeze(tiled_img.shape[0])
           
            img_reconstruct = raw_image.reshape(raw_image.shape[0] % sz, sz*sz, raw_image.shape[1] * sz, 3)
            #img_reconstructed = img_reconstructed_dxdy.reshape(img_reconstructed_dxdy.shape[0],
                                                               #img_reconstructed_dxdy.shape[1], 3)
            tile_reconstruct = tiled_img.reshape(tiled_img.shape[0] % sz, sz*sz, tiled_img.shape[1] * sz, sz%sz)
            #tile_reconstructed = tile_reconstructed_dxdy.reshape(tile_reconstructed_dxdy.shape[0],
                                                                 #tile_reconstructed_dxdy.shape[1], 3)

            print("shape of image after reconstruction:: ", img_reconstruct.shape)
            print("shape of mask after reconstruction:: ", tile_reconstruct.shape)

        return img_reconstruct

# convert 2D array to rle
def reconstruct_to_rle():
    starts = np.where((mask_reconstructed[:-1] == 0) & (mask_reconstructed[1:] == 1))[0]
    ends = np.where((mask_reconstructed[:-1] == 1) & (mask_reconstructed[1:] == 0))[0]
    rle = np.zeros(2 * len(starts))
    rle[::2] = starts
    rle[1::2] = ends - starts
    rle = rle.astype(int)
    return rle


smalllist = [
    "Blur",
    "CenterCrop",
    "HorizontalFlip",
    "VerticalFlip",
    "Normalize",
    "Transpose",
    "RandomGamma",
    "OpticalDistortion",
    "GridDistortion",
    "RandomGridShuffle",
    "HueSaturationValue",
    "PadIfNeeded",
    "RGBShift",
    "RandomBrightness",
    "RandomContrast",
    "MotionBlur",
    "MedianBlur",
    "GaussianBlur",
    "GaussNoise",
    "GlassBlur",
    "CLAHE",
    "ChannelShuffle",
    "ToGray",
    "ToSepia",
    "JpegCompression",
    "ImageCompression",
    "Cutout",
    "CoarseDropout",
    "ToFloat",
    "FromFloat",
    "RandomBrightnessContrast",
    "RandomSnow",
    "RandomRain",
    "RandomFog",
    "RandomSunFlare",
    "RandomShadow",
    "Lambda",
    "ChannelDropout",
    "ISONoise",
    "Solarize",
    "Equalize",
    "Posterize",
    "Downscale",
    "MultiplicativeNoise",
    "FancyPCA",
    "MaskDropout",
    "GridDropout",
    "ColorJitter",
    "ElasticTransform",
    "CropNonEmptyMaskIfExists",
    "IAAAffine",       
    "IAACropAndPad",       
    "IAAFliplr",  
    "IAAFlipud",    
    "IAAPerspective",  
    "IAAPiecewiseAffine",
    "LongestMaxSize",  
    "NoOp",
    "RandomCrop",
    "RandomResizedCrop",
    "RandomScale",
    "RandomSizedCrop",
    "Resize",
    "Rotate",
    "ShiftScaleRotate",
    "SmallestMaxSize"
]

from albumentations import (
 Blur,
 CenterCrop,
 HorizontalFlip,
 VerticalFlip,
 Normalize,
 Transpose,
 RandomGamma,
 OpticalDistortion,
 GridDistortion,
 RandomGridShuffle,
 HueSaturationValue,
 PadIfNeeded,
 RGBShift,
 RandomBrightness,
 RandomContrast,
 MotionBlur,
 MedianBlur,
 GaussianBlur,
 GaussNoise,
 GlassBlur,
 CLAHE,
 ChannelShuffle,
 ToGray,
 ToSepia,
 JpegCompression,
 ImageCompression,
 Cutout,
 CoarseDropout,
 ToFloat,
 FromFloat,
 RandomBrightnessContrast,
 RandomSnow,
 RandomRain,
 RandomFog,
 RandomSunFlare,
 RandomShadow,
 Lambda,
 ChannelDropout,
 ISONoise,
 Solarize,
 Equalize,
 Posterize,
 Downscale,
 MultiplicativeNoise,
 FancyPCA,
 MaskDropout,
 GridDropout,
 ColorJitter,
 ElasticTransform,
 CropNonEmptyMaskIfExists,
 IAAAffine,       
 IAACropAndPad,       
 IAAFliplr,  
 IAAFlipud,    
 IAAPerspective,  
 IAAPiecewiseAffine,
 LongestMaxSize,  
 NoOp,
 RandomCrop,
 RandomResizedCrop,
 RandomScale,
 RandomSizedCrop,
 Resize,
 Rotate,
 ShiftScaleRotate,
 SmallestMaxSize 
)

import albumentations
import random


def transform(image, mask, image_name,mask_name): 
    

    x, y = image, mask
    
    rand = random.uniform(0, 1)
    if(rand > 0.5):
        
        images_name = [f"{image_name}"]
        masks_name = [f"{mask_name}"]
        images_aug = [x]
        masks_aug = [y]

        it = iter(images_name)
        it2 = iter(images_aug)
        imagedict = dict(zip(it, it2))
        
        it = iter(masks_name)
        it2 = iter(masks_aug)
        masksdict = dict(zip(it, it2))
        
        return imagedict, masksdict
    
    mask_density = np.count_nonzero(y)

        ## Augmenting only images with Gloms
    if(mask_density>0):
        try:
            h, w, c = x.shape
        except Exception as e:
            image = image[:-1]
            x, y = image, mask
            h, w, c = x.shape

        aug = Blur(p=1, blur_limit = 3)
        augmented = aug(image=x, mask=y)
        x0 = augmented['image']
        y0 = augmented['mask']
        
    #    aug = CenterCrop(p=1, height=32, width=32)
    #    augmented = aug(image=x, mask=y)
    #    x1 = augmented['image']
    #    y1 = augmented['mask']
            
            ## Horizontal Flip
        aug = HorizontalFlip(p=1)
        augmented = aug(image=x, mask=y)
        x2 = augmented['image']
        y2 = augmented['mask']
            
        aug = VerticalFlip(p=1)
        augmented = aug(image=x, mask=y)
        x3 = augmented['image']
        y3 = augmented['mask']
            
      #      aug = Normalize(p=1)
      #      augmented = aug(image=x, mask=y)
      #      x4 = augmented['image']
      #      y4 = augmented['mask']
            
        aug = Transpose(p=1)
        augmented = aug(image=x, mask=y)
        x5 = augmented['image']
        y5 = augmented['mask']
            
        aug = RandomGamma(p=1)
        augmented = aug(image=x, mask=y)
        x6 = augmented['image']
        y6 = augmented['mask']
            
            ## Optical Distortion
        aug = OpticalDistortion(p=1, distort_limit=2, shift_limit=0.5)
        augmented = aug(image=x, mask=y)
        x7 = augmented['image']
        y7 = augmented['mask']

            ## Grid Distortion
        aug = GridDistortion(p=1)
        augmented = aug(image=x, mask=y)
        x8 = augmented['image']
        y8 = augmented['mask']
            
        aug = RandomGridShuffle(p=1)
        augmented = aug(image=x, mask=y)
        x9 = augmented['image']
        y9 = augmented['mask']  
            
        aug = HueSaturationValue(p=1)
        augmented = aug(image=x, mask=y)
        x10 = augmented['image']
        y10 = augmented['mask']
            
#        aug = PadIfNeeded(p=1)
#        augmented = aug(image=x, mask=y)
#        x11 = augmented['image']
#        y11 = augmented['mask'] 
            
        aug = RGBShift(p=1)
        augmented = aug(image=x, mask=y)
        x12 = augmented['image']
        y12 = augmented['mask']
            
            ## Random Brightness 
        aug = RandomBrightness(p=1)
        augmented = aug(image=x, mask=y)
        x13 = augmented['image']
        y13 = augmented['mask']
            
            ## Random  Contrast
        aug = RandomContrast(p=1)
        augmented = aug(image=x, mask=y)
        x14 = augmented['image']
        y14 = augmented['mask']
            
        #aug = MotionBlur(p=1)
        #augmented = aug(image=x, mask=y)
         #   x15 = augmented['image']
          #  y15 = augmented['mask']
            
        aug = MedianBlur(p=1, blur_limit=5)
        augmented = aug(image=x, mask=y)
        x16 = augmented['image']
        y16 = augmented['mask']
            
        aug = GaussianBlur(p=1, blur_limit=3)
        augmented = aug(image=x, mask=y)
        x17 = augmented['image']
        y17 = augmented['mask']
            
        aug = GaussNoise(p=1)
        augmented = aug(image=x, mask=y)
        x18 = augmented['image']
        y18 = augmented['mask']
        
                    
        aug = GlassBlur(p=1)
        augmented = aug(image=x, mask=y)
        x19 = augmented['image']
        y19 = augmented['mask']
            
        aug = CLAHE(clip_limit=1.0, tile_grid_size=(8, 8), always_apply=False, p=1)
        augmented = aug(image=x, mask=y)
        x20 = augmented['image']
        y20 = augmented['mask']
            
        aug = ChannelShuffle(p=1)
        augmented = aug(image=x, mask=y)
        x21 = augmented['image']
        y21 = augmented['mask']
            
        aug = ToGray(p=1)
        augmented = aug(image=x, mask=y)
        x22 = augmented['image']
        y22 = augmented['mask']
                      
        aug = ToSepia(p=1)
        augmented = aug(image=x, mask=y)
        x23 = augmented['image']
        y23 = augmented['mask']
            
        aug = JpegCompression(p=1)
        augmented = aug(image=x, mask=y)
        x24 = augmented['image']
        y24 = augmented['mask']
            
        aug = ImageCompression(p=1)
        augmented = aug(image=x, mask=y)
        x25 = augmented['image']
        y25 = augmented['mask']
            
        aug = Cutout(p=1)
        augmented = aug(image=x, mask=y)
        x26 = augmented['image']
        y26 = augmented['mask']
            
 #       aug = CoarseDropout(p=1, max_holes=8, max_height=32, max_width=32)
 #       augmented = aug(image=x, mask=y)
 #       x27 = augmented['image']
 #       y27 = augmented['mask']
            
 #       aug = ToFloat(p=1)
 #       augmented = aug(image=x, mask=y)
 #       x28 = augmented['image']
 #       y28 = augmented['mask']
            
        aug = FromFloat(p=1)
        augmented = aug(image=x, mask=y)
        x29 = augmented['image']
        y29 = augmented['mask']
            
            ## Random Brightness and Contrast
        aug = RandomBrightnessContrast(p=1)
        augmented = aug(image=x, mask=y)
        x30 = augmented['image']
        y30 = augmented['mask']
            
        aug = RandomSnow(p=1)
        augmented = aug(image=x, mask=y)
        x31 = augmented['image']
        y31 = augmented['mask']
            
        aug = RandomRain(p=1)
        augmented = aug(image=x, mask=y)
        x32 = augmented['image']
        y32 = augmented['mask']
            
        aug = RandomFog(p=1)
        augmented = aug(image=x, mask=y)
        x33 = augmented['image']
        y33 = augmented['mask']
            
        aug = RandomSunFlare(p=1)
        augmented = aug(image=x, mask=y)
        x34 = augmented['image']
        y34 = augmented['mask']
            
        aug = RandomShadow(p=1)
        augmented = aug(image=x, mask=y)
        x35 = augmented['image']
        y35 = augmented['mask']
            
        aug = Lambda(p=1)
        augmented = aug(image=x, mask=y)
        x36 = augmented['image']
        y36 = augmented['mask']
            
        aug = ChannelDropout(p=1)
        augmented = aug(image=x, mask=y)
        x37 = augmented['image']
        y37 = augmented['mask']
            
        aug = ISONoise(p=1)
        augmented = aug(image=x, mask=y)
        x38 = augmented['image']
        y38 = augmented['mask']
            
        aug = Solarize(p=1)
        augmented = aug(image=x, mask=y)
        x39 = augmented['image']
        y39 = augmented['mask']
            
        aug = Equalize(p=1)
        augmented = aug(image=x, mask=y)
        x40 = augmented['image']
        y40 = augmented['mask']
            
        aug = Posterize(p=1)
        augmented = aug(image=x, mask=y)
        x41 = augmented['image']
        y41 = augmented['mask']
            
        aug = Downscale(p=1)
        augmented = aug(image=x, mask=y)
        x42 = augmented['image']
        y42 = augmented['mask']
            
        aug = MultiplicativeNoise(p=1)
        augmented = aug(image=x, mask=y)
        x43 = augmented['image']
        y43 = augmented['mask']
            
        aug = FancyPCA(p=1)
        augmented = aug(image=x, mask=y)
        x44 = augmented['image']
        y44 = augmented['mask']
            
 #       aug = MaskDropout(p=1)
 #       augmented = aug(image=x, mask=y)
 #       x45 = augmented['image']
 #       y45 = augmented['mask']
            
        aug = GridDropout(p=1)
        augmented = aug(image=x, mask=y)
        x46 = augmented['image']
        y46 = augmented['mask']
            
        aug = ColorJitter(p=1)
        augmented = aug(image=x, mask=y)
        x47 = augmented['image']
        y47 = augmented['mask']
            
            ## ElasticTransform
        aug = ElasticTransform(p=1, alpha=120, sigma=512*0.05, alpha_affine=512*0.03)
        augmented = aug(image=x, mask=y)
        x50 = augmented['image']
        y50 = augmented['mask']
            
        aug = CropNonEmptyMaskIfExists(p=1, height=22, width=32)
        augmented = aug(image=x, mask=y)
        x51 = augmented['image']
        y51 = augmented['mask']

        aug = IAAAffine(p=1)
        augmented = aug(image=x, mask=y)
        x52 = augmented['image']
        y52 = augmented['mask']
            
#        aug = IAACropAndPad(p=1)
#        augmented = aug(image=x, mask=y)
#        x53 = augmented['image']
#        y53 = augmented['mask']
            
        aug = IAAFliplr(p=1)
        augmented = aug(image=x, mask=y)
        x54 = augmented['image']
        y54 = augmented['mask']
            
        aug = IAAFlipud(p=1)
        augmented = aug(image=x, mask=y)
        x55 = augmented['image']
        y55 = augmented['mask']
            
        aug = IAAPerspective(p=1)
        augmented = aug(image=x, mask=y)
        x56 = augmented['image']
        y56 = augmented['mask']
            
        aug = IAAPiecewiseAffine(p=1)
        augmented = aug(image=x, mask=y)
        x57 = augmented['image']
        y57 = augmented['mask']
            
        aug = LongestMaxSize(p=1)
        augmented = aug(image=x, mask=y)
        x58 = augmented['image']
        y58 = augmented['mask']
            
        aug = NoOp(p=1)
        augmented = aug(image=x, mask=y)
        x59 = augmented['image']
        y59 = augmented['mask']
            
            
 #       aug = RandomCrop(p=1, height=22, width=22)
 #       augmented = aug(image=x, mask=y)
 #       x61 = augmented['image']
 #       y61 = augmented['mask']
            
  #      aug = RandomResizedCrop(p=1, height=22, width=20)
  #      augmented = aug(image=x, mask=y)
  #      x63 = augmented['image']
  #      y63 = augmented['mask']
            
        aug = RandomScale(p=1)
        augmented = aug(image=x, mask=y)
        x64 = augmented['image']
        y64 = augmented['mask']
    
            
  #      aug = RandomSizedCrop(p=1, height=22, width=20, min_max_height = [32,32])
  #      augmented = aug(image=x, mask=y)
  #      x66 = augmented['image']
  #      y66 = augmented['mask']
            
  #      aug = Resize(p=1, height=22, width=20)
  #      augmented = aug(image=x, mask=y)
  #      x67 = augmented['image']
  #      y67 = augmented['mask']
            
        aug = Rotate(p=1)
        augmented = aug(image=x, mask=y)
        x68 = augmented['image']
        y68 = augmented['mask']
            
        aug = ShiftScaleRotate(p=1)
        augmented = aug(image=x, mask=y)
        x69 = augmented['image']
        y69 = augmented['mask']
            
        aug = SmallestMaxSize(p=1)
        augmented = aug(image=x, mask=y)
        x70 = augmented['image']
        y70 = augmented['mask']

        images_aug.extend([
            x, x0, x2, x3, x5, x6,
            x7, x8, x9, x10, x12,
            x13, x14, x16, x17, x18, x19, x20,x21, x22,
            x23, x24, x25, x26, x29, x30,x31, x32, x33, x34, x35, x36,
            x37, x38, x39, x40,x41, x42, x43, x44, x46,
            x47, x50,x51, x52, x54, x55, x56,
            x57, x58, x59, x64,
            x68, x69, x70])

        masks_aug.extend([
            y, y0, y2, y3, y5, y6,
            y7, y8, y9, y10, y12,
            y13, y14, y16, y17, y18, y19, y20, y21, y22, y23, y24, y25, y26,
            y29, y30, y31, y32, y33, y34, y35, y36,
            y37, y38, y39, y40, y41, y42, y43, y44, y46,
            y47, y50, y51, y52, y54, y55, y56,
            y57, y58, y59, y64,
            y68, y69, y70])

        idx = -1
        images_name = []
        masks_name = []
        for i, m in zip(images_aug, masks_aug):
            if idx == -1:
                tmp_image_name = f"{image_name}"
                tmp_mask_name  = f"{mask_name}"
            else:
                tmp_image_name = f"{image_name}_{smalllist[idx]}"
                tmp_mask_name  = f"{mask_name}_{smalllist[idx]}"
            images_name.extend(tmp_image_name)
            masks_name.extend(tmp_mask_name)
            idx += 1

        it = iter(images_name)
        it2 = iter(images_aug)
        imagedict = dict(zip(it, it2))
        
        it = iter(masks_name)
        it2 = iter(masks_aug)
        masksdict = dict(zip(it, it2))

    return imagedict, masksdict
