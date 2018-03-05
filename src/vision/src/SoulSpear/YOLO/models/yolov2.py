import torch
import torch.nn as nn
import torch.nn.functional as F
from .loss.region_loss import RegionLoss
from .layers import Route, Reorg, conv

__all__ = ['YOLOv2']


class YOLOv2(nn.Module):
    def __init__(self, anchors, num_classes, width, height):
        super(YOLOv2, self).__init__()

        self.num_classes = num_classes
        self.anchors = anchors
        self.num_anchors = len(anchors)//2
        assert len(anchors) % 2 == 0, "the anchor cfg do not contain full information, a single anchor should contain a h and x pair"
        self.num_output = (num_classes + 5) * self.num_anchors

        self.loss = RegionLoss(self.num_classes, self.anchors, self.num_anchors)

        self.width = width
        self.height = height
        self.erudite = 0

        self.base_cnn = nn.Sequential(
            conv( 3, 32, kernel_size=3, stride=1, padding=1),
            nn.MaxPool2d(2,stride=2),

            conv( 32, 64, kernel_size=3, stride=1, padding=1),
            nn.MaxPool2d(2,stride=2),

            conv( 64, 128, kernel_size=3, stride=1, padding=1),
            conv( 128, 64, kernel_size=1, stride=1, padding=0),
            conv( 64, 128, kernel_size=3, stride=1, padding=1),
            nn.MaxPool2d(2,stride=2),

            conv( 128, 256, kernel_size=3, stride=1, padding=1),
            conv( 256, 128, kernel_size=1, stride=1, padding=0),
            conv( 128, 256, kernel_size=3, stride=1, padding=1),
            nn.MaxPool2d(2,stride=2),

            conv( 256, 512, kernel_size=3, stride=1, padding=1),
            conv( 512, 256, kernel_size=1, stride=1, padding=0),
            conv( 256, 512, kernel_size=3, stride=1, padding=1),
            conv( 512, 256, kernel_size=1, stride=1, padding=0),
            conv( 256, 512, kernel_size=3, stride=1, padding=1),
            )

        self.downsample = nn.MaxPool2d(2,stride=2)
        self.lr_cnn = nn.Sequential(
            conv( 512, 1024, kernel_size=3, stride=1, padding=1),
            conv( 1024, 512, kernel_size=1, stride=1, padding=0),
            conv( 512, 1024, kernel_size=3, stride=1, padding=1),
            conv( 1024, 512, kernel_size=1, stride=1, padding=0),
            conv( 512, 1024, kernel_size=3, stride=1, padding=1),
            conv( 1024, 1024, kernel_size=3, stride=1, padding=1),
            conv( 1024, 1024, kernel_size=3, stride=1, padding=1),
            )
        self.hr_cnn = nn.Sequential(
            conv( 512, 64, kernel_size=1, stride=1, padding=0),
            Reorg(stride=2),
            )
        self.out_cnn = nn.Sequential(
            conv( 1024+64*4, 1024, kernel_size=3, stride=1, padding=1),
            nn.Conv2d( 1024, self.num_output, kernel_size=1, stride=1, padding=1),
            )

    def conv(self, in_channel,out_channel, kernel_size, stride, padding):
        return nn.Sequential(
            nn.Conv2d(in_channel, out_channel, kernel_size=kernel_size, stride=stride, padding=padding, bias=False),
            nn.BatchNorm2d(out_channel),
            nn.LeakyReLU(negative_slope=0.1, inplace=False)
            )

    def loss_setup(self, cfg ):
        loss.object_scale = float(cfg['object_scale'])
        loss.noobject_scale = float(cfg['noobject_scale'])
        loss.class_scale = float(cfg['class_scale'])
        loss.coord_scale = float(cfg['coord_scale'])

    def forward(self,x):

        hr_x = self.base_cnn(x)

        lr_x = self.downsample(hr_x) # reduce by half
        lr_x = self.lr_cnn(lr_x)

        hr_x = self.hr_cnn(hr_x)

        x = torch.cat( [lr_x, hr_x] , dim=1)
        x = self.out_cnn(x)
        return x

    def print_network(self):
        print(self)
