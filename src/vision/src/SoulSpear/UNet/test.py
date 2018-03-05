from __future__ import print_function
import sys
#if len(sys.argv) != 4:
#    print('Usage:')
#    print('python train.py datacfg cfgfile weightfile')
#    exit()
#import random
#import math
import os
import time
import argparse
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torch.backends.cudnn as cudnn
from torchvision import datasets, transforms
from torch.autograd import Variable
from tensorboardX import SummaryWriter
from tqdm import tqdm

import utils.dataset as dataset
from utils.utils import bbox_iou, nms, get_region_boxes
from utils.utils import logging, file_lines
#from region_loss import RegionLoss
#from darknet import Darknet
#from models.tiny_yolo import TinyYoloNet
#import models
from cfg.cfg import parse_cfg
from utils.timer import Timer
from utils.logger import Logger
from utils.model_io import load_checkpoint, save_checkpoint, create_portfolio, load_model
from load import load_model_method, load_cfg_name

assert torch.cuda.is_available() == True, "check cuda"

model_names, models = load_model_method()
data_cfgs , model_cfgs = load_cfg_name()

parser = argparse.ArgumentParser(description="Train")
parser.add_argument(
    '--arch','-a',
    help="architecture",
    type=str,
    choices=model_names)
parser.add_argument(
    '--data','-d',
    help="datasource",
    type=str,
    choices=data_cfgs)

parser.add_argument('--debug', help="debug mode", type=bool)

args = parser.parse_args()

cfg = parse_cfg( "arch/" + args.arch )
data_cfg = parse_cfg( "data/" + args.data )
hardware = parse_cfg( 'hardware' )

# Training settings
checkpoint    = cfg["checkpoint"]
tensorboard   = cfg['tensorboard']
train_options = cfg["train"]
thres         = cfg["threshold"]

testlist      = data_cfg['valid']

#nsamples      = file_lines(trainlist)
gpus          = hardware['gpus']  # e.g. 0,1,2,3
ngpus         = len(gpus)
num_workers   = int( hardware['ncore'] )

batch_size    = int( train_options['batch_size'] )

tf_dir        = os.path.expanduser( tensorboard['root'] )
summary_interval = train_options['summary_interval']
#Train parameters



cp_path       = checkpoint['path']
cp_best       = checkpoint['best']

max_epochs    = 1
use_cuda      = True
seed          = int(time.time())
eps           = 1e-5
save_interval = train_options['save_interval']  # epoches
dot_interval  = 70  # batches

# Test parameters
conf_thresh   = thres['conf_thresh']
nms_thresh    = thres['nms_thresh']
iou_thresh    = thres['iou_thresh']

###############
torch.manual_seed(seed)
if use_cuda:
    gpus = ','.join( [ str(i) for i in gpus ] )
    os.environ['CUDA_VISIBLE_DEVICES'] = gpus
    torch.cuda.manual_seed(seed)

model       = models[ args.arch ]( **cfg['model'] )
region_loss = model.loss

global_step = 0 # record how many training step it has taken
generation = 0 # record how many time this model got throwing into training
load_model( model, cp_path, cp_best ,which = 1)
#model.print_network()

region_loss.seen  = model.erudite
processed_batches = model.erudite/batch_size

init_width        = model.width
init_height       = model.height

kwargs = {'num_workers': num_workers, 'pin_memory': True} if use_cuda else {}
test_loader = torch.utils.data.DataLoader(
    dataset.listDataset(testlist, shape=(init_width, init_height),
                   shuffle=False,
                   transform=transforms.Compose([
                       transforms.ToTensor(),
                   ]), train=False),
    batch_size=batch_size, shuffle=False, **kwargs)

if use_cuda:
    if ngpus > 1:
        model = torch.nn.DataParallel(model).cuda()
    else:
        model = model.cuda()

def test( writer ):
    def truths_length(truths):
        for i in range( 50 ):
            if np.array_equal( truths[i], np.zeros(5) ) :
                return i

    timer = Timer()

    model.eval()
    if ngpus > 1:
        cur_model = model.module
    else:
        cur_model = model

    num_classes = cur_model.num_classes
    anchors     = cur_model.anchors
    num_anchors = cur_model.num_anchors
    total       = 0.0
    proposals   = 0.0
    correct     = 0.0
    after_nms   = 0.0

    pbar = tqdm( total= len(test_loader) )
    for batch_idx, (data, target) in enumerate(test_loader):

        pbar.update(1)
        #timer.event("start")
        if use_cuda:
            data = data.cuda()
        #timer.event('to gpu')
        data = Variable(data, volatile=True)
        output = model(data).data
        #timer.event('YOLO model')

        all_boxes = get_region_boxes(output, conf_thresh, num_classes, anchors, num_anchors)
        #timer.event('get region boxes')

        for i in range(output.size(0)):
            boxes = all_boxes[i]
            boxes = nms(boxes, nms_thresh)

            truths = target[i].view(-1, 5)
            num_gts = truths_length(truths)

            after_nms += len(boxes)
            total = total + num_gts

            for i in range(len(boxes)):
                if boxes[i][4] > conf_thresh:
                    proposals = proposals+1

            for i in range(num_gts):
                box_gt = [truths[i][1], truths[i][2], truths[i][3], truths[i][4], 1.0, 1.0, truths[i][0]]
                best_iou = 0
                best_j = -1
                for j in range(len(boxes)):
                    iou = bbox_iou(box_gt, boxes[j], x1y1x2y2=False)
                    if iou > best_iou:
                        best_j = j
                        best_iou = iou
                if best_iou > iou_thresh and boxes[best_j][6] == box_gt[6]:
                    correct = correct+1
        #timer.event('evaluate')
        #timer.calculate()
        #timer.report(mode='rec',release=True)

    pbar.close()
    precision = 1.0*correct/(proposals+eps)
    recall = 1.0*correct/(total+eps)
    fscore = 2.0*precision*recall/(precision+recall+eps)
    logging("precision: %f, recall: %f, fscore: %f, boxes: %f, nGT: %f" % (precision, recall, fscore, after_nms, total ))

logging('evaluating ...')
writer = SummaryWriter( log_dir = tf_dir )
test( writer )
