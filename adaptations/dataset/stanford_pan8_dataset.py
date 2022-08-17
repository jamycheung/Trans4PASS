import os
import json
import os.path as osp
import torch
from torchvision import transforms
import numpy as np
import torchvision
from PIL import Image
from torch.utils import data
import glob
from utils.transform import FixScaleRandomCropWH, FixScaleRandomCropWH_joint, RandomScaleCrop_joint
from dataset.stanford_pin_dataset import __FOLD__

import torch.nn.functional as F
from imageio import imread

class StanfordPan8DataSet(data.Dataset):
    def __init__(self, root, list_path, max_iters=None, crop_size=(321, 321), 
                 mean=(128, 128, 128), scale=True, mirror=True, ignore_label=0,
                 set='val', ssl_dir='', trans='FixScaleRandomCropWH',
                 fold=1):
        self.root = root
        # self.list_path = list_path
        self.crop_size = crop_size
        self.scale = scale
        self.ignore_label = ignore_label
        self.mean = mean
        self.is_mirror = mirror
        self.ssl_dir = ssl_dir
        self.img_paths = _get_stanford2d3d_path(root, fold, set)
        if not max_iters==None:
            self.img_paths = self.img_paths * int(np.ceil(float(max_iters) / len(self.img_paths)))
        self.files = []
        self.set = set
        self.trans = trans
        for p in self.img_paths:
            self.files.append({
                "img": p,
                "name": p.split(self.root+'/')[-1]
            })

    def __len__(self):
        return len(self.files)

    def __getitem__(self, index):
        datafiles = self.files[index]

        image = Image.open(datafiles["img"]).convert('RGB')
        name = datafiles["name"]
        # --- crop top, bottom black area
        width, height = image.size
        left, top, right, bottom = 0, 320, width, 1728
        image = image.crop((left, top, right, bottom))
        image = image.resize((width,height), Image.BILINEAR)
        if self.trans == 'resize':
            # resize
            image = image.resize(self.crop_size, Image.BICUBIC)
        elif self.trans == 'FixScaleRandomCropWH':
            # resize, keep ratio
            image = FixScaleRandomCropWH(self.crop_size, is_label=False)(image)
        else:
            raise NotImplementedError

        size = np.asarray(image, np.float32).shape

        input_transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((.485, .456, .406), (.229, .224, .225)),
        ])
        image = input_transform(image)

        if len(self.ssl_dir) > 0:
            label = Image.open(osp.join(self.ssl_dir, name.replace('.png', '_labelTrainIds.png')))
            # label = Image.open(osp.join(self.ssl_dir, name))
            label = label.crop((left, top, right, bottom))
            label = label.resize((width,height), Image.NEAREST)

            if self.trans == 'resize':
                # resize
                label = label.resize(self.crop_size, Image.NEAREST)
            elif self.trans == 'FixScaleRandomCropWH':
                # resize, keep ratio
                label = FixScaleRandomCropWH(self.crop_size, is_label=True)(label)
            else:
                raise NotImplementedError
            label = torch.LongTensor(np.array(label).astype('int32'))
            return image, label, np.array(size), name

        return image, np.array(size), name


class StanfordPan8TestDataSet(data.Dataset):
    def __init__(self, root, list_path, max_iters=None, crop_size=(4096, 2048), mean=(128, 128, 128),
                scale=False, mirror=False, ignore_label=255, set='val', fold=1, trans='resize'):
        self.root = root
        self.crop_size = crop_size
        self.img_paths = _get_stanford2d3d_path(root, fold, set)

        if not max_iters==None:
            self.img_paths = self.img_paths * int(np.ceil(float(max_iters) / len(self.img_paths)))
        self.files = []
        # --- stanford color2id
        with open('dataset/s2d3d_pin_list/semantic_labels.json') as f:
            id2name = [name.split('_')[0] for name in json.load(f)] + ['<UNK>']
        with open('dataset/s2d3d_pin_list/name2label.json') as f:
            name2id = json.load(f)
        self.colors = np.load('dataset/s2d3d_pin_list/colors.npy') # for visualization
        self.id2label = np.array([name2id[name] for name in id2name], np.uint8)
        self.trans = trans

        for p in self.img_paths:
            self.files.append({
                "img": p,
                "label": p.replace("rgb", "semantic"),
                "name": p.split(self.root+'/')[-1]
            })
        self._key = np.array([255,255,255,0,1,255,255,2,3,4,5,6,7])

    def __len__(self):
        return len(self.files)

    def _map13to8(self, mask):
        values = np.unique(mask)
        for value in values:
            if value == 255: 
                mask[mask==value] = 255
            else:
                mask[mask==value] = self._key[value]
        return mask

    def _color2id(self, img, sem):
        sem = np.array(sem, np.int32)
        rgb = np.array(img, np.int32)
        unk = (sem[..., 0] != 0)
        sem = self.id2label[sem[..., 1] * 256 + sem[..., 2]]
        sem[unk] = 0
        sem[rgb.sum(-1) == 0] = 0
        sem -= 1 # 0->255
        return Image.fromarray(sem)

    def __getitem__(self, index):
        datafiles = self.files[index]


        image = Image.open(datafiles["img"]).convert('RGB')

        label = Image.open(datafiles["label"])
        label = self._color2id(image, label)
        label = self._map13to8(np.array(label).astype('int32'))
        label = Image.fromarray(label)
        name = datafiles["name"]

        # resize
        if self.trans == 'resize':
            # resize
            image = image.resize(self.crop_size, Image.BICUBIC)
            label = label.resize(self.crop_size, Image.NEAREST)
        elif self.trans == 'FixScaleRandomCropWH_joint':
            # resize, keep ratio
            image, label = FixScaleRandomCropWH_joint(self.crop_size)(image, label)
        elif self.trans == 'RandomScaleCrop_joint':
            # resize, keep ratio
            image, label = RandomScaleCrop_joint()(image, label)
        else:
            raise NotImplementedError

        size = np.asarray(image).shape
        input_transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((.485, .456, .406), (.229, .224, .225)),
        ])
        image = input_transform(image)
        label = torch.LongTensor(np.array(label).astype('int32'))
        # print(image.shape, label.shape)

        return image, label, np.array(size), name

def _get_stanford2d3d_path(folder, fold, mode='train'):
    '''image is jpg, label is png'''
    img_paths = []
    if mode == 'train':
        area_ids = __FOLD__['{}_{}'.format(fold, mode)]
    elif mode == 'val':
        area_ids = __FOLD__['{}_{}'.format(fold, mode)]
    elif mode == 'trainval':
        area_ids = __FOLD__[mode]
    else:
        raise NotImplementedError
    for a in area_ids:
        img_paths += glob.glob(os.path.join(folder, '{}/pano/rgb/*_rgb.png'.format(a)))
    img_paths = sorted(img_paths)
    return img_paths


if __name__ == '__main__':
    dst = StanfordPan8DataSet("data/Stanford2D3D", 'dataset/s2d3d_pan_list/val.txt', mean=(0,0,0))
    trainloader = data.DataLoader(dst, batch_size=4)
    for i, data in enumerate(trainloader):
        imgs, labels, _ = data
        if i == 0:
            img = torchvision.utils.make_grid(imgs).numpy()
            img = np.transpose(img, (1, 2, 0))
            img = img[:, :, ::-1]
            img = Image.fromarray(np.uint8(img) )
            img.show()
        break
