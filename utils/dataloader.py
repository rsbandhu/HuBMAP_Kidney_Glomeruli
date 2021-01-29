import numpy as np
from skimage import io, transform
import torch 

from torchvision import transforms, utils
from torch.utils.data import Dataset, DataLoader
import os
import pickle

import matplotlib.pyplot as plt
#import preprocessing_utils as utils
from utils import preprocessing_utils as utils

class Dataset_Image_mask(Dataset):

    def __init__(self, data_dir, mean, std_dev, transform=None):
        super(Dataset_Image_mask, self).__init__()
        self.root_dir = data_dir
        self.transform = transform
        self.mask_dict = self.get_mask_dict()
        self.img_name_list = list(self.mask_dict.keys())
        self.len = self.__len__()
        self.normalize = transforms.Normalize(mean, std_dev)

        

    def get_mask_dict(self):
        '''
        open the pickled file containing the dict of mask in rle format
        returns
        dict: key same as imgae file name
        value: numpy array in rle format
        '''
        img_and_mask_files = os.listdir(self.root_dir)
        mask_rle_file = [x for x in img_and_mask_files if "mask" in x][0]
        mask_rle_dict = pickle.load(
            open(os.path.join(self.root_dir, mask_rle_file), 'rb'))
        return mask_rle_dict

    def __len__(self):
        return len(self.img_name_list)

    def __getitem__(self, idx):
        img_name = self.img_name_list[idx]
        img_file_name = os.path.join(self.root_dir, f"{img_name}.png")
        img = io.imread(img_file_name)
        #img = torch.tensor(img.transpose((2,0,1))).float()
        #img = img.transpose(2, 0, 1)
        
        mask_rle_np = self.mask_dict[img_name]
        #print("mask rle shape :: ", mask_rle_np.shape)
        #print(f"**\n{mask_rle_np}")
        mask_rle = " ".join(str(x) for x in mask_rle_np)
        
        mask_2d = utils.mask_rle_to_2d(mask_rle, 512, 512)
        #augment image for training
        if self.transform is not None:
            mask_name = f"mask_{img_name}"
            img, mask_2d = self.transform(img, mask_2d, img_name, mask_name)
            #mask_2d = self.transform(mask_2d)
        
        img = self.normalize(transforms.ToTensor()(img))
        mask = torch.from_numpy(mask_2d).long()

        sample = {'image': img, 'mask': mask, 'idx': img_name}
        #sample = {'image': img}
        return sample

def main():
    data_dir = '/media/bony/Ganga_HDD_3TB/Ganges_Backup/Machine_Learning/HuBMAP_Hacking_Kidney/hubmap-kidney-segmentation/train/tiled_thresholded_512'
    mean = [0.68912, 0.47454, 0.6486]
    std_dev = [0.13275, 0.23647, 0.15536]


    dataset = Dataset_Image_mask(data_dir, mean, std_dev)
    n_tot = dataset.len

    train_test_split = 0.8
    train_count = int(train_test_split * n_tot)

    test_count = dataset.len - train_count

    train_idx = list(np.random.choice(range(n_tot), train_count, replace = False))
    test_idx = list(set(range(n_tot)) - set(train_idx))

    print(len(train_idx), len(test_idx), n_tot - len(train_idx) - len(test_idx))

    train_ds = torch.utils.data.Subset(dataset, train_idx)
    test_ds = torch.utils.data.Subset(dataset, test_idx)

    dataloader = DataLoader(dataset, batch_size=1, shuffle=True, num_workers=0)
    train_loader = DataLoader(train_ds, batch_size=1, shuffle=False, num_workers=0)
    test_loader = DataLoader(test_ds, batch_size= 4, shuffle= True, num_workers= 0)
    
    for i, sample_batch in enumerate(test_loader):
        print(i, sample_batch['image'].shape, sample_batch['mask'].shape)

    

    '''
    for i in range(30):
        sample = dataset[i]
        print(sample['image'].shape)

        if i == 83:
            img = sample['image']
            plt.imshow(img)
    '''

if __name__ == '__main__':
    main()


