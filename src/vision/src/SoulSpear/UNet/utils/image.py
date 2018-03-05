import numpy as np
from PIL import Image

def data_augmentation(img, shape, aug):
    return img

def load_data_label(imgpath, labpath, shape, aug=None):
    # img
    img = Image.open(imgpath)
    if img.mode == "RGBA":
        img = img.convert("RGB")
    if aug is not None:
        img = data_augmentation(img, aug)
    img = img.resize(shape)
    #print(np.array(img).shape)
    # mask
    if labpath is None :
        mask = np.zeros(shape)
    else:
        mask = Image.open(labpath)
        mask = mask.resize(shape)
        mask = np.array(mask)
        #print('mask: ',mask.shape)
        #print(labpath)
        #mask = mask[:,:,0]
        mask[mask == 255] = 1
        mask = np.expand_dims(mask,0)
        #mask = np.expand_dims(mask,0)

    return img, mask

def load_data_listlabel(imgpath, labpath, shape, aug=None, merge=False):
    # img
    img = Image.open(imgpath)
    if img.mode == "RGBA":
        img = img.convert("RGB")
    if aug is not None:
        img = data_augmentation(img, aug)
    img = img.resize(shape)

    # mask
    if labpath is None :
        mask = np.zeros((1,1))
    else:
        masks = list()
        for path in labpath:
            mask = np.array(Image.open(labpath))
            mask[mask==255] = 1
            masks.append(mask[:,:,0])
        mask = np.stack(masks, dim=0)
        if merge:
             mask = np.sum(mask, dim=0, keepdims=True)

    return img, mask
