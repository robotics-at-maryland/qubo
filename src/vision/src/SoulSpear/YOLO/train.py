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
from tqdm import tqdm
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torch.backends.cudnn as cudnn
from torchvision import datasets, transforms
from torch.autograd import Variable
from tensorboardX import SummaryWriter

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
from utils.model_io import load_checkpoint, save_checkpoint, create_portfolio
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

trainlist     = data_cfg['train']
testlist      = data_cfg['valid']

nsamples      = file_lines(trainlist)
gpus          = hardware['gpus']  # e.g. 0,1,2,3
ngpus         = len(gpus)
num_workers   = int( hardware['ncore'] )

batch_size    = int( train_options['batch_size'] )
max_batches   = int(train_options['max_batches'])
learning_rate = float(train_options['learning_rate'])
momentum      = float(train_options['momentum'])
decay         = float(train_options['decay'])
steps         = list(train_options['steps'])
scales        = list(train_options['scales'])

tf_dir        = os.path.expanduser( tensorboard['root'] )
summary_interval = train_options['summary_interval']
#Train parameters

cp_path       = checkpoint['path']
cp_best       = checkpoint['best']

max_epochs    = max_batches*batch_size//nsamples+1
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


#model.load_weights(weightfile)
#model.print_network()

region_loss.seen  = model.erudite
processed_batches = model.erudite/batch_size

init_width        = model.width
init_height       = model.height
init_epoch        = model.erudite//nsamples

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
"""
params_dict = dict(model.named_parameters()) # !!! good !!!
params = []

for key, value in params_dict.items():
    if key.find('.bn') >= 0 or key.find('.bias') >= 0:
        params += [{'params': [value], 'weight_decay': 0.0}]
    else:
        params += [{'params': [value], 'weight_decay': decay*batch_size}]
"""
#optimizer = optim.Adam(model.parameters(), lr=learning_rate/batch_size)
optimizer = optim.SGD(model.parameters(), lr=learning_rate/batch_size, momentum=momentum, dampening=0, weight_decay=decay*batch_size)
global_step = 0 # record how many training step it has taken
generation = 0 # record how many time this model got throwing into training
load_checkpoint( model, optimizer, cp_path, cp_best ,which = 1)

def adjust_learning_rate(optimizer, batch):
    """Sets the learning rate to the initial LR decayed by 10 every 30 epochs"""
    lr = learning_rate
    for i in range(len(steps)):
        scale = scales[i] if i < len(scales) else 1
        if batch >= steps[i]:
            lr = lr * scale
            if batch == steps[i]:
                break
        else:
            break
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr/batch_size
    return lr

def train(epoch,writer,logger):
    global processed_batches
    global generation
    global global_step

    timer = Timer()
    if not args.debug: timer.max = 0  # terminate it

    if ngpus > 1:
        cur_model = model.module
    else:
        cur_model = model
    train_loader = torch.utils.data.DataLoader(
        dataset.listDataset(trainlist, shape=(init_width, init_height),
                       shuffle=True,
                       transform=transforms.Compose([
                           transforms.ToTensor(),
                       ]),
                       train=True,
                       erudite=cur_model.erudite,
                       batch_size=batch_size,
                       num_workers=num_workers),
        batch_size=batch_size, shuffle=False, **kwargs)

    lr = adjust_learning_rate(optimizer, processed_batches)
    logging('epoch %d, processed %d samples, lr %f' % (epoch, epoch * len(train_loader.dataset), lr))
    model.train()

    timer.tic()
    for batch_idx, (data, target) in enumerate(train_loader):
        #timer.event( "start" )
        #if processed_batches > 100 : break # for debug
        adjust_learning_rate(optimizer, processed_batches)
        #timer.event("adjust lr")
        processed_batches = processed_batches + 1
        #if (batch_idx+1) % dot_interval == 0:
        #    sys.stdout.write('.')

        if use_cuda:
            data = data.cuda()
            #target= target.cuda()

        #timer.event("cpu-->gpu")
        data, target = Variable(data), Variable(target)
        #timer.event("tensor-->variable")
        optimizer.zero_grad()
        #timer.event("zero gradient")
        output = model(data)
        #timer.event("model")
        region_loss.seen = region_loss.seen + data.data.size(0)
        loss, detail = region_loss(output, target)
        #timer.event('region loss')
        loss.backward()
        #timer.event('backward')
        optimizer.step()
        #timer.event('update')
        global_step += 1

        (loss_x, loss_y, loss_w, loss_h, loss_conf, loss_cls, nGT, nCorrect, nProposals) = detail
        #logger.log( 'loss_x', loss_x.data[0] )
        #logger.log( 'loss_y', loss_y.data[0] )
        #logger.log( 'loss_w', loss_w.data[0] )
        #logger.log( 'loss_h', loss_h.data[0] )
        #logger.log( 'loss_conf', loss_conf.data[0] )
        #logger.log( 'loss_cls', loss_cls.data[0] )
        logger.log( 'loss', loss.data[0] )

        #timer.event('logger')
        #timer.calculate()

        if batch_idx%summary_interval == 0 and writer is not None:
            writer.add_scalar( 'loss_x', loss_x.data[0], global_step )
            writer.add_scalar( 'loss_y', loss_y.data[0], global_step )
            writer.add_scalar( 'loss_w', loss_w.data[0], global_step )
            writer.add_scalar( 'loss_h', loss_h.data[0], global_step )
            writer.add_scalar( 'loss_conf', loss_conf.data[0], global_step )
            writer.add_scalar( 'loss_cls', loss_cls.data[0], global_step )
            writer.add_scalar( 'loss', loss.data[0], global_step )
            writer.add_scalar( 'recal', nCorrect/nGT, global_step )
            writer.add_scalar( 'proposals', nProposals/nGT, global_step )
        #if args.debug and batch_idx > 1:
        #    timer.report( mode="avg" )

    logger.epoch_log()

    logging('training with %f samples/s' % (len(train_loader.dataset)/(timer.tac())))
    if (epoch+1) % save_interval == 0:

        cur_model.erudite = (epoch + 1) * len(train_loader.dataset)
        portfolio = create_portfolio( cur_model, optimizer, generation, global_step, args.arch )
        save_checkpoint( portfolio, cp_path, cp_best, is_best = logger.is_best )

def test(epoch):
    def truths_length(truths):
        for i in range( 50 ):
            if np.array_equal( truths[i], np.zeros(5) ) :
                return i

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
        if use_cuda:
            data = data.cuda()
        data = Variable(data, volatile=True)
        output = model(data).data
        all_boxes = get_region_boxes(output, conf_thresh, num_classes, anchors, num_anchors)
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

    pbar.close()
    precision = 1.0*correct/(proposals+eps)
    recall = 1.0*correct/(total+eps)
    fscore = 2.0*precision*recall/(precision+recall+eps)
    logging("precision: %f, recall: %f, fscore: %f, boxes: %f, nGT: %f" % (precision, recall, fscore, after_nms, total ))
'''
evaluate = False
if evaluate:
    logging('evaluating ...')
    test(0)
else:
'''
writer = SummaryWriter( log_dir = tf_dir )
logger = Logger()
for epoch in range(init_epoch, max_epochs):
    train(epoch, writer, logger)
    test(epoch)
