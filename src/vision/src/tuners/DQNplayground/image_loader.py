
#CV2 format B G R [H W C]
#Torch format [C H W]

#http://pytorch.org/tutorials/beginner/data_loading_tutorial.html

from __future__ import print_function, division
import sys
if (sys.version_info > (3, 0) ):
    import queue
else:
    import Queue as queue


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
#import argparse

import threading
import time

# https://pymotw.com/3/queue/


class CV2videoloader(threading.Thread):
    def __init__(self,name,videosrc,video_buffer,pulse=0.5,camera_config=None):
        self.videosrc= videosrc if videosrc != None else 0
        self.camera = cv2.VideoCapture(self.videosrc)
        self.buffer = video_buffer if isinstance( video_buffer , queue.Queue) else None
        self.stop = False
        self.pulse = pulse
        self.debug = False
        threading.Thread.__init__(self)

    def run(self):
        while not self.stop:
            time.sleep(self.pulse)
            if self.debug: print("----get one----")
            (grabbed, frame) = self.camera.read()
            if grabbed == False:
                warnings.warn("the camera {} is grabing air".format(self.videosrc) )
            else:
                self.buffer.put(frame)

class torchStream():
    def __init__(self,name,video_buffer,batchsize=1,transform=None):

        self.name =name
        self.video_buffer= video_buffer if isinstance( video_buffer , queue.Queue) else None

        if transform!= None:
            self.transform = transform
        else:
             self.transform=transforms.ToTensor()
             warnings.warn("at least do a ToTensor operation to the raw image")

        self.batchsize = batchsize
        self.batchbuffer = list()

    def __len__(self):
        return self.video_buffer.qsize()

    def batchpop(self):
        self.batchbuffer = [ self.transform( self.video_buffer.get() ) for i in range(self.batchsize) ]
        return torch.stack( self.batchbuffer,0 )

    def pop(self):
        return self.transform( self.video_buffer.get() )

class tfstream():
    def __init__(self,name,video_buffer):
        pass

    def __call__(self):
        return 0

    def stop():
        return 0

    def isstop():
        return 0

if __name__ == '__main__':


    data_transform = transforms.Compose([
            transforms.ToPILImage(),
            transforms.Scale(256),
            transforms.ToTensor(),
                                ])

    def show_stream_batch(sample_batched):
        images_batch = sample_batched
        batch_size = len(images_batch)
        grid = utils.make_grid(images_batch)
        t=transforms.ToPILImage()
        plt.imshow( t(grid) )



    video_buffer = queue.Queue( maxsize = 30 )
    CV2L= CV2videoloader( name='test',
            videosrc=0,video_buffer=video_buffer,
            camera_config=None )

    torS = torchStream(name='test',video_buffer=video_buffer,batchsize=5,transform=data_transform )

    CV2L.start()
    time.sleep(3)
    show_stream_batch( torS.batchpop() )
    CV2L.stop = True
    CV2L.join()
    plt.show()
