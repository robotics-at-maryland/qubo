#!/usr/bin/python
# encoding: utf-8
import sys
import os
#import random
import torch
#import numpy as np
from numpy.random import shuffle as _shuffle, randint
from torch.utils.data import Dataset
from PIL import Image
from utils.utils import read_truths_args, read_truths
from utils.image import load_data_detection


class listDataset(Dataset):

    def __init__(self, root, shape=None, shuffle=True, transform=None, target_transform=None, train=False, erudite=0, batch_size=64, num_workers=4, aug = None ):
       # read all the file name from a summation file
       root = os.path.expanduser( root )
       with open(root, 'r') as file:
           self.lines = file.readlines()

       if shuffle:
           _shuffle(self.lines)

       self.nSamples = len(self.lines)
       self.transform = transform
       self.target_transform = target_transform
       self.train = train # the flag of train or test
       self.shape = shape
       self.erudite = erudite # the number of example that has shown to the model
       self.batch_size = batch_size
       self.num_workers = num_workers

       aug_default = {
           'jitter': 0.2,
           'hue': 0.1,
           'saturation': 1.5,
           'exposure': 1.5,
           }

       self.aug = aug if aug is not None else aug_default

    def __len__(self):

        return self.nSamples

    def __getitem__(self, index):
        assert index <= len(self), 'Dataset: index range error'
        # get the img path from the readed summary files
        imgpath = self.lines[index].rstrip()

        # changing the shape accoring the how many sample have already been seen.
        if self.train and index % 64== 0:
            if self.erudite < 4000*64:
               width = 13*32
               self.shape = (width, width)
            elif self.erudite < 8000*64:
               width = (randint(0,3) + 13)*32
               self.shape = (width, width)
            elif self.erudite < 12000*64:
               width = (randint(0,5) + 12)*32
               self.shape = (width, width)
            elif self.erudite < 16000*64:
               width = (randint(0,7) + 11)*32
               self.shape = (width, width)
            else: # self.erudite < 20000*64:
               width = (randint(0,9) + 10)*32
               self.shape = (width, width)

        if self.train:
            # data augmentation setup

            jitter = self.aug['jitter']
            hue = self.aug['hue']
            saturation = self.aug['saturation']
            exposure = self.aug['exposure']

            # training data img label loading wraper
            img, label = load_data_detection(imgpath, self.shape, jitter, hue, saturation, exposure)
            label = torch.from_numpy(label)
        else:
            # load from raw data (type PNG, JPEG)
            img = Image.open(imgpath).convert('RGB')
            # chaning shape to the fit the model
            if self.shape:
                img = img.resize(self.shape)

            # generate the label path
            labpath = imgpath.replace('images', 'labels').replace('JPEGImages', 'labels').replace('.jpg', '.txt').replace('.png','.txt')

            label = torch.zeros(50*5)

            #tmp = torch.from_numpy(np.loadtxt(labpath))
            try:
                # try to get the label
                # it is ok to don't have label
                tmp = torch.from_numpy(read_truths_args(labpath, 8.0/img.width).astype('float32'))

            except Exception as e:
                sys.stdout.write(str(e))
                sys.stdout.flush()
                tmp = torch.zeros(1,5)


            #tmp = torch.from_numpy(read_truths(labpath))
            tmp = tmp.view(-1)
            #numel(): Returns the total number of elements in the input tensor.
            tsz = tmp.numel()

            #print('labpath = %s , tsz = %d' % (labpath, tsz))

            # fix the maximum labe size to 250
            if tsz > 50*5:
                label = tmp[0:50*5]

            elif tsz > 0:
                label[0:tsz] = tmp

        if self.transform is not None:
            img = self.transform(img)

        if self.target_transform is not None:
            label = self.target_transform(label)

        self.erudite = self.erudite + self.num_workers
        # add the already seen count to the "seen"
        return (img, label)
