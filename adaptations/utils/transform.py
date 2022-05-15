
import numpy as np
import torch
import numbers
import random
from PIL import Image,ImageFilter,ImageOps

def colormap_cityscapes(n):
    cmap=np.zeros([n, 3]).astype(np.uint8)
    cmap[0,:] = np.array([128, 64,128])
    cmap[1,:] = np.array([244, 35,232])
    cmap[2,:] = np.array([ 70, 70, 70])
    cmap[3,:] = np.array([ 102,102,156])
    cmap[4,:] = np.array([ 190,153,153])
    cmap[5,:] = np.array([ 153,153,153])

    cmap[6,:] = np.array([ 250,170, 30])
    cmap[7,:] = np.array([ 220,220,  0])
    cmap[8,:] = np.array([ 107,142, 35])
    cmap[9,:] = np.array([ 152,251,152])
    cmap[10,:] = np.array([ 70,130,180])

    cmap[11,:] = np.array([ 220, 20, 60])
    cmap[12,:] = np.array([ 255,  0,  0])
    cmap[13,:] = np.array([ 0,  0,142])
    cmap[14,:] = np.array([  0,  0, 70])
    cmap[15,:] = np.array([  0, 60,100])

    cmap[16,:] = np.array([  0, 80,100])
    cmap[17,:] = np.array([  0,  0,230])
    cmap[18,:] = np.array([ 119, 11, 32])
    cmap[19,:] = np.array([ 0,  0,  0])
    
    return cmap


def colormap(n):
    cmap=np.zeros([n, 3]).astype(np.uint8)

    for i in np.arange(n):
        r, g, b = np.zeros(3)

        for j in np.arange(8):
            r = r + (1<<(7-j))*((i&(1<<(3*j))) >> (3*j))
            g = g + (1<<(7-j))*((i&(1<<(3*j+1))) >> (3*j+1))
            b = b + (1<<(7-j))*((i&(1<<(3*j+2))) >> (3*j+2))

        cmap[i,:] = np.array([r, g, b])

    return cmap

class Relabel:

    def __init__(self, olabel, nlabel):
        self.olabel = olabel
        self.nlabel = nlabel

    def __call__(self, tensor):
        assert isinstance(tensor, torch.LongTensor) or isinstance(tensor, torch.ByteTensor) , 'tensor needs to be LongTensor'
        tensor[tensor == self.olabel] = self.nlabel
        return tensor


class ToLabel:

    def __call__(self, image):
        return torch.from_numpy(np.array(image)).long()
        
        #return np.asarray(image, np.float32)


class Colorize:

    def __init__(self, n=22):
        #self.cmap = colormap(256)
        self.cmap = colormap_cityscapes(256)
        self.cmap[n] = self.cmap[-1]
        self.cmap = torch.from_numpy(self.cmap[:n])

    def __call__(self, gray_image):
        size = gray_image.size()
        color_image = torch.ByteTensor(3, size[1], size[2]).fill_(0)

        #for label in range(1, len(self.cmap)):
        for label in range(0, len(self.cmap)):
            mask = gray_image[0] == label

            color_image[0][mask] = self.cmap[label][0]
            color_image[1][mask] = self.cmap[label][1]
            color_image[2][mask] = self.cmap[label][2]

        return color_image


class RandomCrop(object):
    def __init__(self, crop_size, is_label=False):
        if isinstance(crop_size, numbers.Number):
            self.crop_size = (int(crop_size), int(crop_size))
        else:
            self.crop_size = crop_size
        self.is_label = is_label
        self.base_size = 0

    def __call__(self, sample):
        w, h = sample.size
        # random crop crop_size
        x1 = random.randint(0, w - self.crop_size[1])
        y1 = random.randint(0, h - self.crop_size[0])
        if self.is_label:
            sample = sample.crop((x1, y1, x1 + self.crop_size[1], y1 + self.crop_size[0]))
        else:
            sample = sample.crop((x1, y1, x1 + self.crop_size[1], y1 + self.crop_size[0]))
        return sample


class RandomHorizontalFlip(object):
    def __call__(self, sample):
        if random.random() < 0.5:
            sample = sample.transpose(Image.FLIP_LEFT_RIGHT)
        return sample


class RandomRotate(object):
    def __init__(self, degree, is_label=False):
        self.degree = degree
        self.is_label = is_label
    def __call__(self, sample):
        rotate_degree = random.uniform(-1*self.degree, self.degree)
        sample = sample.rotate(rotate_degree, Image.BILINEAR)
        if self.is_label:
            sample = sample.rotate(rotate_degree, Image.NEAREST)
        return sample


class RandomGaussianBlur(object):
    def __call__(self, sample):
        if random.random() < 0.5:
            sample = sample.filter(ImageFilter.GaussianBlur(
                radius=random.random()))
        return sample


class RandomScaleCrop(object):
    def __init__(self, base_size, crop_size, fill=0, is_label=False):
        self.base_size = base_size
        self.crop_size = crop_size
        self.fill = fill
        self.is_label = is_label

    def __call__(self, sample):
        # img = sample['image']
        # mask = sample['label']
        # random scale (short edge)
        short_size = random.randint(int(self.base_size * 0.5), int(self.base_size * 2.0))
        w, h = sample.size
        if h > w:
            ow = short_size
            oh = int(1.0 * h * ow / w)
        else:
            oh = short_size
            ow = int(1.0 * w * oh / h)
        sample = sample.resize((ow, oh), Image.NEAREST if self.is_label else Image.BILINEAR)
        # pad crop
        if short_size < self.crop_size:
            padh = self.crop_size - oh if oh < self.crop_size else 0
            padw = self.crop_size - ow if ow < self.crop_size else 0
            sample = ImageOps.expand(sample, border=(0, 0, padw, padh), fill=0)
        # random crop crop_size
        w, h = sample.size
        x1 = random.randint(0, w - self.crop_size)
        y1 = random.randint(0, h - self.crop_size)
        sample = sample.crop((x1, y1, x1 + self.crop_size, y1 + self.crop_size))
        return sample

class RandomScaleCrop_joint(object):
    def __init__(self, base_size=2048, crop_size=1080, fill=255):
        self.base_size = base_size
        self.crop_size = crop_size
        self.fill = fill

    def __call__(self, img, lb):
        # img = sample['image']
        # mask = sample['label']
        # random scale (short edge)
        if random.random() < 0.5:
            img = img.transpose(Image.FLIP_LEFT_RIGHT)
            lb = lb.transpose(Image.FLIP_LEFT_RIGHT)
        if random.random() < 0.5:
            img = img.filter(ImageFilter.GaussianBlur(radius=random.random()))
        short_size = random.randint(int(self.base_size * 0.5), int(self.base_size * 2.0))
        w, h = img.size
        if h > w:
            ow = short_size
            oh = int(1.0 * h * ow / w)
        else:
            oh = short_size
            ow = int(1.0 * w * oh / h)
        img = img.resize((ow, oh), Image.BILINEAR)
        lb = lb.resize((ow, oh), Image.NEAREST)
        # pad crop
        if short_size < self.crop_size:
            padh = self.crop_size - oh if oh < self.crop_size else 0
            padw = self.crop_size - ow if ow < self.crop_size else 0
            img = ImageOps.expand(img, border=(0, 0, padw, padh), fill=self.fill)
            lb = ImageOps.expand(lb, border=(0, 0, padw, padh), fill=self.fill)
        # random crop crop_size
        w, h = img.size
        x1 = random.randint(0, w - self.crop_size)
        y1 = random.randint(0, h - self.crop_size)
        img = img.crop((x1, y1, x1 + self.crop_size, y1 + self.crop_size))
        lb = lb.crop((x1, y1, x1 + self.crop_size, y1 + self.crop_size))
        return img, lb

class FixScaleCropWH_Center(object):
    def __init__(self, crop_size_wh, is_label=False):
        assert isinstance(crop_size_wh, tuple)
        self.crop_size = crop_size_wh
        self.is_label = is_label

    def __call__(self, sample):
        w, h = sample.size
        cw, ch = self.crop_size
        if w < cw:
            # new w h
            nw = cw
            nh = int(1.0 * h * nw / w)
            sample = sample.resize((nw, nh), Image.NEAREST if self.is_label else Image.BILINEAR)
        w, h = sample.size
        if h < ch:
            # new w h
            nh = ch
            nw = int(1.0 * nh * w / h)
            sample = sample.resize((nw, nh), Image.NEAREST if self.is_label else Image.BILINEAR)

        # center crop
        w, h = sample.size
        x1 = int(round((w - self.crop_size[0]) / 2.))
        y1 = int(round((h - self.crop_size[1]) / 2.))
        # left, upper, right, and lower
        sample = sample.crop((x1, y1, x1 + self.crop_size[0], y1 + self.crop_size[1]))
        return sample


class FixScaleRandomCropWH(object):
    def __init__(self, crop_size_wh, is_label=False):
        assert isinstance(crop_size_wh, tuple)
        self.crop_size = crop_size_wh
        self.is_label = is_label

    def __call__(self, sample):
        w, h = sample.size
        cw, ch = self.crop_size
        if w < cw:
            # new w h
            nw = cw
            nh = int(1.0 * h * nw / w)
            sample = sample.resize((nw, nh), Image.NEAREST if self.is_label else Image.BILINEAR)
        w, h = sample.size
        if h < ch:
            # new w h
            nh = ch
            nw = int(1.0 * nh * w / h)
            sample = sample.resize((nw, nh), Image.NEAREST if self.is_label else Image.BILINEAR)

        # # center crop
        # w, h = sample.size
        # x1 = int(round((w - self.crop_size[0]) / 2.))
        # y1 = int(round((h - self.crop_size[1]) / 2.))
        # random crop crop_size
        w, h = sample.size
        x1 = random.randint(0, w - self.crop_size[0])
        y1 = random.randint(0, h - self.crop_size[1])
        # left, upper, right, and lower
        sample = sample.crop((x1, y1, x1 + self.crop_size[0], y1 + self.crop_size[1]))
        return sample

class FixScaleRandomCropWH_joint(object):
    def __init__(self, crop_size_wh):
        assert isinstance(crop_size_wh, tuple)
        self.crop_size = crop_size_wh

    def __call__(self, img, mask):
        w, h = img.size
        cw, ch = self.crop_size
        if w < cw:
            # new w h
            nw = cw
            nh = int(1.0 * h * nw / w)
            img = img.resize((nw, nh), Image.BILINEAR)
            mask = mask.resize((nw, nh), Image.NEAREST)
        w, h = img.size
        if h < ch:
            # new w h
            nh = ch
            nw = int(1.0 * nh * w / h)
            img = img.resize((nw, nh), Image.BILINEAR)
            mask = mask.resize((nw, nh), Image.NEAREST)

        # random crop crop_size
        w, h = img.size
        x1 = random.randint(0, w - self.crop_size[0])
        y1 = random.randint(0, h - self.crop_size[1])
        # left, upper, right, and lower
        img = img.crop((x1, y1, x1 + self.crop_size[0], y1 + self.crop_size[1]))
        mask = mask.crop((x1, y1, x1 + self.crop_size[0], y1 + self.crop_size[1]))
        return img, mask

class FixScaleCrop(object):
    def __init__(self, crop_size, is_label=False):
        self.crop_size = crop_size
        self.is_label = is_label

    def __call__(self, sample):
        w, h = sample.size
        if w > h:
            oh = self.crop_size
            ow = int(1.0 * w * oh / h)
        else:
            ow = self.crop_size
            oh = int(1.0 * h * ow / w)
        sample = sample.resize((ow, oh), Image.NEAREST if self.is_label else Image.BILINEAR)

        # center crop
        w, h = sample.size
        x1 = int(round((w - self.crop_size) / 2.))
        y1 = int(round((h - self.crop_size) / 2.))
        sample = sample.crop((x1, y1, x1 + self.crop_size, y1 + self.crop_size))
        return sample

class FixedResize(object):
    def __init__(self, size, is_label=False):
        self.size = (size, size)  # size: (h, w)
        self.is_label = is_label

    def __call__(self, sample):
        sample = sample.resize(self.size, Image.NEAREST if self.is_label else Image.BILINEAR)
        return sample