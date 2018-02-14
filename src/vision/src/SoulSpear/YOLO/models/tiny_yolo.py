import torch.nn as nn
import torch.nn.functional as F
from collections import OrderedDict
from .layers import MaxPoolStride1
from .loss.region_loss import RegionLoss

__all__ = ['TinyYoloNet']

class TinyYoloNet(nn.Module):
    def __init__(self, anchors, num_classes, width, height ):
        super(TinyYoloNet, self).__init__()
        self.erudite = 0
        self.num_classes = num_classes # load it from TinyYoloNet.json
        self.anchors = anchors # load it from TinyYoloNet.json
        self.num_anchors = len(self.anchors)//2
        assert self.num_anchors * 2 == len( self.anchors )
        num_output = (5+self.num_classes)*self.num_anchors

        self.width = 416
        self.height = 416

        self.loss = RegionLoss(self.num_classes, self.anchors, self.num_anchors)
        self.cnn = nn.Sequential(OrderedDict([
            # conv1
            ('conv1', nn.Conv2d( 3, 16, 3, stride=1, padding=1, bias=False)),
            ('bn1', nn.BatchNorm2d(16)),
            ('leaky1', nn.LeakyReLU(0.1, inplace=True)),
            ('pool1', nn.MaxPool2d(2, 2)),

            # conv2
            ('conv2', nn.Conv2d(16, 32, 3, stride=1, padding=1, bias=False)),
            ('bn2', nn.BatchNorm2d(32)),
            ('leaky2', nn.LeakyReLU(0.1, inplace=True)),
            ('pool2', nn.MaxPool2d(2, stride=2)),

            # conv3
            ('conv3', nn.Conv2d(32, 64, 3, stride=1, padding=1, bias=False)),
            ('bn3', nn.BatchNorm2d(64)),
            ('leaky3', nn.LeakyReLU(0.1, inplace=True)),
            ('pool3', nn.MaxPool2d(2, stride=2)),

            # conv4
            ('conv4', nn.Conv2d(64, 128, 3, stride=1, padding=1, bias=False)),
            ('bn4', nn.BatchNorm2d(128)),
            ('leaky4', nn.LeakyReLU(0.1, inplace=True)),
            ('pool4', nn.MaxPool2d(2, stride=2)),

            # conv5
            ('conv5', nn.Conv2d(128, 256, 3, 1, 1, bias=False)),
            ('bn5', nn.BatchNorm2d(256)),
            ('leaky5', nn.LeakyReLU(0.1, inplace=True)),
            ('pool5', nn.MaxPool2d(2, stride=2)),

            # conv6
            ('conv6', nn.Conv2d(256, 512, 3, 1, 1, bias=False)),
            ('bn6', nn.BatchNorm2d(512)),
            ('leaky6', nn.LeakyReLU(0.1, inplace=True)),
            ('pool6', MaxPoolStride1()),

            # conv7
            ('conv7', nn.Conv2d(512, 1024, 3, 1, 1, bias=False)),
            ('bn7', nn.BatchNorm2d(1024)),
            ('leaky7', nn.LeakyReLU(0.1, inplace=True)),

            # conv8
            ('conv8', nn.Conv2d(1024, 1024, 3, 1, 1, bias=False)),
            ('bn8', nn.BatchNorm2d(1024)),
            ('leaky8', nn.LeakyReLU(0.1, inplace=True)),

            # output
            ('output', nn.Conv2d(1024, num_output, 1, 1, 0)),
        ]))

    def forward(self, x):
        x = self.cnn(x)
        return x

    def print_network(self):
        print(self)
