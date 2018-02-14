
from __future__ import print_function



import sys


import os
import time
import argparse
import numpy as np
from tqdm import tqdm
from PIL import Image, ImageDraw

import torch
from torchvision import transforms
from torch.autograd import Variable

from utils.utils import do_detect
from utils.utils import  plot_boxes, plot_boxes_cv2, plot_boxes_plt
from utils.utils import load_class_names

from cfg.cfg import parse_cfg
from utils.timer import Timer
#from utils.logger import Logger
from utils.model_io import load_model
from load import load_model_method, load_cfg_name


model_names, models = load_model_method()
data_cfgs , model_cfgs = load_cfg_name()

parser = argparse.ArgumentParser(description="Detect ... ")
parser.add_argument(
    '--arch','-a',
    help="architecture",
    type=str,
    choices=model_names
    )
parser.add_argument(
    '--data',
    help="the dataset the model trained on",
    type=str,
    choices=data_cfgs
)
parser.add_argument(
    '--dir','-d',
    nargs = "+",
    help="data you want to detect",
    type=str,
    )

parser.add_argument(
    '--fr',
    help="the output frame rate for video detector",
    type=int,
    default=40
)

parser.add_argument('--debug', help="debug mode", type=bool)

args = parser.parse_args()

cfg = parse_cfg( "arch/" + args.arch )
data_cfg = parse_cfg( "data/" + args.data )
hardware = parse_cfg( 'hardware' )

class_names_proxy = data_cfg['class_names']
class_names = load_class_names(class_names_proxy)

# Training settings
checkpoint    = cfg["checkpoint"]
tensorboard   = cfg['tensorboard']
train_options = cfg["train"]
thres         = cfg["threshold"]

gpus          = hardware['gpus']  # e.g. 0,1,2,3
ngpus         = len(gpus)
num_workers   = int( hardware['ncore'] )

tf_dir        = os.path.expanduser( tensorboard['root'] )
summary_interval = train_options['summary_interval']
#Train parameters

cp_path       = checkpoint['path']
cp_best       = checkpoint['best']

use_cuda      = True
seed          = int(time.time())

# Test parameters
conf_thresh   = thres['conf_thresh']
nms_thresh    = thres['nms_thresh']
iou_thresh    = thres['iou_thresh']

FRAME_RATE = args.fr

###############
torch.manual_seed(seed)
if use_cuda:
    gpus = ','.join( [ str(i) for i in gpus ] )
    os.environ['CUDA_VISIBLE_DEVICES'] = gpus
    torch.cuda.manual_seed(seed)

model       = models[ args.arch ]( **cfg['model'] )


if use_cuda:
    if ngpus > 1:
        model = torch.nn.DataParallel(model).cuda()
    else:
        model = model.cuda()

global_step = 0 # record how many training step it has taken
generation = 0 # record how many time this model got throwing into training
load_model( model, cp_path, cp_best ,which = 1)

def detect( imgfile, savename ):
    assert model.num_classes == len(class_names), "might attach a wrong data config to the model"
    if use_cuda: model.cuda()

    img = Image.open(imgfile).convert('RGB')
    sized = img.resize((model.width, model.height))

    start = time.time()
    boxes = do_detect(model, sized, conf_thresh, nms_thresh, use_cuda)
    finish = time.time()
    print('%s: Predicted in %f seconds.' % (imgfile, (finish-start)))

    plot_boxes(img, boxes, '{}_pred.jpg'.format(savename), class_names)

def detect_cv2(imgfile):
    import cv2

    if use_cuda: model.cuda()

    img = cv2.imread(imgfile)
    sized = cv2.resize(img, (model.width, model.height))
    sized = cv2.cvtColor(sized, cv2.COLOR_BGR2RGB)

    start = time.time()
    boxes = do_detect(model, sized, conf_thresh, nms_thresh, use_cuda)
    finish = time.time()

    print('%s: Predicted in %f seconds.' % (imgfile, (finish-start)))

    class_names = load_class_names(namesfile)
    plot_boxes_cv2(img, boxes, savename='predictions.jpg', class_names=class_names)

def detect_skimage(imgfile):
    from skimage import io
    from skimage.transform import resize

    use_cuda = 1
    if use_cuda:
        model.cuda()

    img = io.imread(imgfile)
    sized = resize(img, (model.width, model.height))

    start = time.time()
    boxes = do_detect(model, sized, conf_thresh, nms_thresh, use_cuda)
    finish = time.time()
    print('%s: Predicted in %f seconds.' % (imgfile, (finish-start)))

    class_names = load_class_names(namesfile)
    plot_boxes_cv2(img, boxes, savename='predictions.jpg', class_names=class_names)

def detect_skvideo( path, name= None ):

    from utils.timer import Timer
    timer = Timer()
    def reset(*pathes_type):
        for patches in pathes_type:
            [patch.remove() for patch in patches ]

    import skvideo.io
    from skimage.transform import resize
    import matplotlib.pyplot as plt
    #import matplotlib.patches as patches

    if use_cuda:model.cuda()
    videogen = skvideo.io.vreader( str( path ) )

    plt.ion()
    fig, ax = plt.subplots( 1,1, figsize = ( 16, 9 ) )
    plt.axis( 'off' )
    plt.tight_layout()

    frame = next( videogen )
    sized = resize(frame, (model.width, model.height), preserve_range=True)

    win = ax.imshow( frame )

    timer.tic()
    for frame in videogen:
        print('load frame ', timer.tac())
        print('background ', timer.tac())
        win.set_data(frame)
        print('set frame ',timer.tac())
        sized = resize(frame, (model.width, model.height), preserve_range=True)
        print('resize', timer.tac())
        boxes = do_detect(model, sized, conf_thresh, nms_thresh, use_cuda )
        print('detect box', timer.tac())
        RecList, TextList= plot_boxes_plt( frame, boxes, ax, savename = None, class_names = class_names )
        print('plot boxes', timer.tac())
        plt.draw()
        print('draw', timer.tac() )
        plt.pause( 1/FRAME_RATE )
        print('pause', timer.tac() )
        list( map( lambda x: x.remove(), RecList) )
        list( map( lambda x: x.remove(), TextList) )
        #reset( RecList, TextList )
        print('reset ', timer.tac() )

    '''         RUN TIME DATA
        load frame  0.0036106109619140625
        set frame  0.04231595993041992
        resize 0.04909849166870117
        detect box 0.020146846771240234
        plot boxes 0.0008680820465087891
        draw 4.100799560546875e-05
        pause 0.05254626274108887
        reset  0.00010919570922851562
    '''

def detect_cv2video( path, name= None ):
    pass
if __name__ == '__main__':
    detect_api = {
     'jpg':detect,
     'png':detect,
     'jpeg':detect,
     'avi':detect_skvideo,
     }

    for filename in args.dir:
        name, suffix = filename.split('.')
        detect_api[suffix](filename, name )
