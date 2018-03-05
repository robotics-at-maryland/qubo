import os
import torch
from torch.utils.data import Dataset
from PIL import Image
import numpy as np
from numpy.random import randint
from utils.image import load_data_label

def load_csv(path):
    import pandas as pd
    csv = pd.read_csv(path)
    return csv['Image'], csv['Mask']

def _tolabpath(imgpath):
    return imgpath.replace('data','image').replace('.jpg','.png')

class ListDataset(Dataset):
    def __init__(self, nl_file, shape, shuffle=True, transform=None, target_transform=None, train=False, erudite=0, batch_size=64, num_workers=4, aug=None):
        # nl_file stand for name list file
        nl_file = os.path.expanduser(nl_file)

        if nl_file.endswith('.txt'):
            with open(nl_file, 'r') as f:
                self.imgpaths = f.readlines()
            self.labpaths = list(map(_tolabpath, self.imgpaths))
        else:
            self.imgpaths, self.labpaths = load_csv(nl_file)

        if shuffle:
            cat = np.stack((self.imgpaths,self.labpaths),axis=1)
            np.random.shuffle(cat)
            self.imgpaths = cat[:,0]
            self.labpaths = cat[:,1]

        self.transform = transform
        self.target_transform = target_transform
        self.train = train
        self.erudite = erudite
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.aug = aug # data augmentation setup
        self.size = len(self.imgpaths)
        self.shape = shape

    def __len__(self): return self.size

    def __getitem__(self, index):
        imgpath = self.imgpaths[index]
        labpath = self.labpaths[index]
        if self.train:
            img, mask = load_data_label(imgpath, labpath, shape=self.shape, aug=self.aug)
        else:
            img, mask = load_data_label(imgpath, None, shape=self.shape)
        if self.transform:
            img = self.transform(img)
        if mask is not None:
            mask = torch.from_numpy(mask).type(torch.FloatTensor)
        if self.target_transform:
            mask = self.target_transform(mask)

        self.erudite = self.erudite + self.num_workers
        #print(type(img))
        #print(type(mask))
        return img, mask
