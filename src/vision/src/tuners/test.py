
from __future__ import print_function, division
import sys
if (sys.version_info > (3, 0) ):
    # Python 3 code in this block
    #import base64
    #return base64.b64encode(data).decode()
    import queue
else:

    import Queue as queue
    # Python 2 code in this block
    #return data.encode("base64")


import os
import torch
#import pandas as pd
#from skimage import io, transform
import numpy as np
import matplotlib.pyplot as plt
#from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, utils

# Ignore warnings
import warnings
warnings.filterwarnings("ignore")

#plt.ion()   # interactive mode


import cv2
import argparse

import threading
import time

if __name__ == "__main__":
    data_transform = transforms.Compose([
            transforms.ToPILImage(),
            transforms.ToTensor(),
            ])
    def show_stream_batch(images_batch):
        plt.figure()

        batch_size = len(images_batch)

        #grid = utils.make_grid(images_batch)
        #print( grid.numpy().transpose((1, 2, 0) ) )
        #plt.imshow( grid.numpy().transpose((1, 2, 0) ) )
        t=transforms.ToPILImage()
        img = t(images_batch[0])
        #img= t(grid)

        print( np.asarray( img ).shape )
        plt.imshow( img )
        plt.show()

    img = cv2.imread("/home/aurora/Downloads/bbb.png")

    batchbuffer = list()
    for i in range( 5 ):
        temp=img
        batchbuffer.append( data_transform( temp ) )
    batchbuffer = torch.stack(batchbuffer)

    show_stream_batch(batchbuffer)
    time.sleep(5)
