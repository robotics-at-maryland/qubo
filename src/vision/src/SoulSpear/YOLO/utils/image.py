#!/usr/bin/python
# encoding: utf-8
import random
import os
from PIL import Image
import numpy as np

def within(x):
    return x>-0.01 and x<1.001

def scale_image_channel(im, c, v):
    cs = list(im.split())
    cs[c] = cs[c].point(lambda i: i * v)
    out = Image.merge(im.mode, tuple(cs))
    return out

def distort_image(im, hue, sat, val):
    im = im.convert('HSV')
    cs = list(im.split())
    cs[1] = cs[1].point(lambda i: i * sat)
    cs[2] = cs[2].point(lambda i: i * val)

    def change_hue(x):
        x += hue*255
        if x > 255:
            x -= 255
        if x < 0:
            x += 255
        return x
    cs[0] = cs[0].point(change_hue)
    im = Image.merge(im.mode, tuple(cs))

    im = im.convert('RGB')
    #constrain_image(im)
    return im

def rand_scale(s):
    scale = random.uniform(1, s)
    if(random.randint(1,10000)%2):
        return scale
    return 1./scale

def random_distort_image(im, hue, saturation, exposure):
    dhue = random.uniform(-hue, hue)
    dsat = rand_scale(saturation)
    dexp = rand_scale(exposure)
    res = distort_image(im, dhue, dsat, dexp)
    return res

def data_augmentation(img, shape, jitter, hue, saturation, exposure):
    # original scale
    oh = img.height
    ow = img.width
    # use cropping factor to calculate the offset limit from the org img
    dw =int(ow*jitter)
    dh =int(oh*jitter)
    # offset value
    pleft  = random.randint(0, dw)
    pright = random.randint(0, dw)
    ptop   = random.randint(0, dh)
    pbot   = random.randint(0, dh)
    # remained size of the image
    swidth =  ow - pleft - pright
    sheight = oh - ptop - pbot
    # index of crop
    sx = ow/float(swidth)
    sy = oh/float(sheight)
    #sx = float(swidth)  / ow
    #sy = float(sheight) / oh
    # flip flag
    flip = random.randint(1,300)%2
    cropped = img.crop( (pleft, ptop, pleft+swidth, ptop+sheight))
    #cropped = img.crop( (pleft, ptop, pleft + swidth - 1, ptop + sheight - 1))
    #relative size of cropped size w.r.t
    dx = float(pleft)/swidth
    dy = float(ptop)/sheight
    #dx = (float(pleft)/ow)/sx
    #dy = (float(ptop) /oh)/sy

    sized = cropped.resize(shape)

    if flip:
        sized = sized.transpose(Image.FLIP_LEFT_RIGHT)
    img = random_distort_image(sized, hue, saturation, exposure)
    assert dx>=0 and dy>=0, (dx,dy)
    assert sx>=0 and sy>=0, (sx,sy)
    return img, flip, dx,dy,sx,sy

def fill_truth_detection(labpath, w, h, flip, dx, dy, sx, sy):
    max_boxes = 50
    label = np.zeros((max_boxes,5))
    if os.path.getsize(labpath):
        bs = np.loadtxt(labpath) # relative value
        if bs is None:
            return label
        bs = np.reshape(bs, (-1, 5))
        cc = 0

        for i in range(bs.shape[0]):
            x1 = bs[i][1] - bs[i][3]/2.0
            y1 = bs[i][2] - bs[i][4]/2.0
            x2 = bs[i][1] + bs[i][3]/2.0
            y2 = bs[i][2] + bs[i][4]/2.0
            assert within(x1) and within(x2) and within(y1) and within(y2), ('rec',x1,x2,y1,y2)

            # recalibrate the box parameter according to the index of crop and
            # rnew = r*ori/(new) - off/(new)
            x1 = x1*sx - dx
            y1 = y1*sy - dy
            x2 = x2*sx - dx
            y2 = y2*sy - dy

            # recalibrate those out of frame bounding box
            x1 = min(0.999, max(0, x1))
            y1 = min(0.999, max(0, y1))
            x2 = min(0.999, max(0, x2))
            y2 = min(0.999, max(0, y2))
            #assert within(x1) and within(x2) and within(y1) and within(y2), ('cab',x1,x2,y1,y2,sx,sy,dx,dy,old)

            bs[i][1] = (x1 + x2)/2.0
            bs[i][2] = (y1 + y2)/2.0
            bs[i][3] = (x2 - x1)
            bs[i][4] = (y2 - y1)

            if flip:
                bs[i][1] =  0.999 - bs[i][1]

            if bs[i][3] < 0.001 or bs[i][4] < 0.001:
                continue
            label[cc] = bs[i]
            cc += 1
            if cc >= 50:
                break

    label = np.reshape(label, (-1))
    return label

def load_data_detection(imgpath, shape, jitter, hue, saturation, exposure):
    labpath = imgpath.replace('images', 'labels').replace('JPEGImages', 'labels').replace('.jpg', '.txt').replace('.png','.txt')

    ## data augmentation
    img = Image.open(imgpath).convert('RGB')
    img,flip,dx,dy,sx,sy = data_augmentation(img, shape, jitter, hue, saturation, exposure)
    label = fill_truth_detection(labpath, img.width, img.height, flip, dx, dy, sx, sy)
    return img,label
