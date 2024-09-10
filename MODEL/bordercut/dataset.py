import random, sys
from pathlib import Path

import torch
import torchvision
from torch.utils.data import IterableDataset, Dataset, DataLoader
import kornia
import numpy as np
import pandas as pd
import os
from sultan.api import Sultan as Bash

PROJ_ROOT = Path(*Path.cwd().parts[:Path().cwd().parts.index('legibility-estimation')+1])

sys.path.append(str(PROJ_ROOT) + '/data')
import border_utils as bu

sys.path.append(str(PROJ_ROOT) + '/MODEL/GMM')
from gmmy_bear import create_tri_mask

class BordersDataset(Dataset):
    def __init__(self, img_paths,same_border, prefix=f'{str(PROJ_ROOT)}/bing_maps/'):
        random.shuffle(img_paths)
        self.prefix = prefix
        self.data, self.masks = self.create_tile_arrays(img_paths)
        self.same_border = same_border
        if self.same_border == True:
            print("Error: Same-border code not implemented, defaulting to False")
            self.same_border = False
        
    # Returns an array of np arrays
    def create_tile_arrays(self, img_paths):
        tile_array = []
        mask_array = []
        h, w, c = None, None, None
        for path in img_paths:
            img = bu.imread(self.prefix + path)
            h,w,c = img.shape
            mask = create_tri_mask(self.prefix+path, (h,w), (h,w), color=2, thick=random.randrange(5,30), show=False)
            try:
                if mask == None:
                    print("ERROR: Unable to create mask for " + path)
            except:
                tile_array.append(img)
                mask_array.append(mask)
        return tile_array, mask_array

    def get_next_tile(self, mixes, i):
        # return tile and mask 
        img = self.data[i]
        mask = self.masks[i]
        
        return self.mix(img, mask, mixes), mask
        
    ''' 
        Where we actually replace image features.
        args:
            img: np.array (shape HxWxC)
            mask: np.array (shape HxW)
            sides: list with a 0, 1, or 2 (e.g [0, 1, 2] or [0, 1] or [1])
                (sides is the same thing as arg 'mixes' in self.get_next_tile())
    '''
    def mix(self, img, mask, sides):
        for side in sides:           
            rand = self.get_random_tile()
            #rand = self.get_random_tile() if not self.same_border else bu.imread(self.from_same_border(tile))
            img[mask == side] = rand[mask == side]
        return img
    
   
    
    # grabs a random border from the same border as tile
    def from_same_border(self, tile):
        random_tile = tile
        country_directory = f"{self.prefix}global-rectified-point4/{Path(tile).parts[-3]}"
        directories = os.listdir(country_directory)

        while(random_tile == tile):
            random_tile_direc = random.choice(directories)
            random_tile = Path(country_directory + "/" + random_tile_direc + "/" + random_tile_direc + ".jpeg")
            random_tile = "/".join(random_tile.parts[-4:])
        return self.prefix + random_tile
        
    # returns a random image array 
    def get_random_tile(self):
        return random.choice(self.data)
    
    def unprep_img(self, x):
        x = kornia.utils.tensor_to_image(x).astype(np.uint8)
        return x
    
    ''' Turns np.array into a model ready torch array with augmentations applied '''
    def prep_img(self, x, rot=True):
        x = kornia.utils.image_to_tensor(x, keepdim=False)
        # (13, 13), (2, 2), (25, 25), (4, 4)
        x = kornia.filters.gaussian_blur2d(x.float(), (25, 25), (4, 4))
        if rot:
            #x = kornia.geometry.transform.rotate(x, torch.randint(0, 360, (1,)).float())
            if random.random() > 0.5:
                x = kornia.geometry.transform.vflip(x)
            if random.random() > 0.5:
                x = kornia.geometry.transform.hflip(x)
        return x.squeeze()

    ''' Same thing as prep_img() but only used if we're training with masks '''
    def prep_img_mask(self, x, mask, rot=True):
        x = kornia.utils.image_to_tensor(x.copy(), keepdim=False)
        x = kornia.filters.gaussian_blur2d(x.float(), (13, 13), (2, 2))
        # convert 0, 1, 2 mask to -1, 0, 1 
        mask = torch.tensor(mask - 1).float()
        x = torch.vstack([x.squeeze(), mask.unsqueeze(0)]) \
                .unsqueeze(0)
        if rot and random.random() > 0.5:
            x = kornia.geometry.transform.vflip(x)
        if rot and random.random() > 0.5:
            x = kornia.geometry.transform.hflip(x)
            
        return x.squeeze()
        
    ''' 
        Decides which sides in each image should be replaced.
        i.e an option [0, 2] means side 0 and side 2 should be replaced.
    '''
    def get_mixes(self):
        #options = [[0,2], [1,3], [0,3]] #Mix More
        #n_mixes1, n_mixes2 = 0, 2 # Mix Same
        
        # Mix Same 2
        options = [
            [random.choice([[0], [1]]), [2]],
            [[], [2]],
            [[], [random.choice([0,1]), 2]]
        ]
        n_mixes1, n_mixes2 = random.choice(options)
        return n_mixes1, n_mixes2
    
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, i):
        ''' Decide how we will mix the images '''
        n_mixes1, n_mixes2 = self.get_mixes()
        mask = None
        while mask is None:
            x1, mask = self.get_next_tile(n_mixes1, i)
        should_be_same_img = False #random.random() < 0.5
        should_flip_x1_x2 = random.random() > 0.5

        y = torch.ones(1).squeeze().long()

        '''Mix and update label y. (0: x1 is more mixed, 1: x2 is more mixed)'''
        if should_be_same_img: 
            x2 = x1.copy()
        else:
            x2 = self.mix(x1.copy(), mask, n_mixes2)
            
        ''' Augment images and transform to tensor '''
        x1 = self.prep_img(x1)
        x2 = self.prep_img(x2)

        # randomly swap on a coin flip to keep us on our toes
        if should_flip_x1_x2:
            x1, x2 = x2, x1
            n_mixes1, n_mixes2 = n_mixes2, n_mixes1
            y = 1 - y

        return x1, x2, y

    ''' This is used instead of __getitem__() when we are training with masks  '''
    def __getitem1__(self, i):
        print("Executing __getitem1__")
        n_mixes1, n_mixes2 = self.get_mixes()
        mask = None
        while mask is None:
            x1, mask, tile = self.get_next_tile(n_mixes1)
        mask2 = mask.copy()
        should_be_same_img = random.random() < 0.7
        should_be_same_mask = False#random.random() < 0.5
        should_rot_mask = random.random() < 0.5
        should_flip_x1_x2 = random.random() > 0.5

        #y = torch.ones(2).squeeze().float()
        y = torch.ones(1).squeeze().long()

        if should_be_same_img: 
            # Rotate x1 and give it random mask 
            if not should_be_same_mask:
                x2 = self.mix(x1, mask, n_mixes2, tile)
                x1 = x2.copy()
                if should_rot_mask:
                    x1 = np.rot90(x1, k=1)
                else:
                    img = self.get_random_tile()
                    h, w, c = img.shape
                    mask = None
                    while mask is None:
                        mask = create_tri_mask(tile, (h,w), (h,w), color=2, 
                                               thick=random.randrange(5, 30), show=False)
                        #mask = create_tri_mask(tile, (h,w), (h,w), color=2, thick=5, show=False)

                #y[0] = 0
            else:
                # random choose to make both x1 or x2
                if random.random() > 0.5:
                    x2 = self.mix(x1, mask, n_mixes2, tile)
                    x1 = x2.copy()
                else:
                    x2 = x1.copy()
                y *= 0.5
                
        else:
            x2 = self.mix(x1.copy(), mask, n_mixes2, tile)
            #y[0] = 0 
            
        x1 = self.prep_img_mask(x1, mask)
        x2 = self.prep_img_mask(x2, mask2)

        self.i += 1
        
        # randomly swap on a coin flip to keep us on our toes
        if should_flip_x1_x2:
            x1, x2 = x2, x1
            n_mixes1, n_mixes2 = n_mixes2, n_mixes1
            y = 1 - y

        
        return x1, x2, y
        
''' Creates and returns model ready train and validation loaders '''
def get_train_and_val_loader(imgs_txt, n, split, batch_size=4, same_border=False, test=False):
    
    all_imgs = bu.shuf_file(imgs_txt, n)

    trainset = BordersDataset(all_imgs[:int(n * split)], same_border=same_border)
    valset = BordersDataset(all_imgs[int(n * split):], same_border=same_border)

    train = DataLoader(trainset, batch_size=batch_size, num_workers=8, drop_last=True)
    val = DataLoader(valset, batch_size=batch_size, num_workers=8, drop_last=True)
    
    return train, val
