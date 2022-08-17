import os.path as osp
import os.path as osp

import numpy as np
import torch
import torchvision
from PIL import Image
from torch.utils import data
from torchvision import transforms

from utils.transform import FixScaleRandomCropWH


class densepass13DataSet(data.Dataset):
    def __init__(self, root, list_path, max_iters=None, crop_size=(1024, 200),
                 mean=(128, 128, 128), scale=True, mirror=True, ignore_label=255, set='val', ssl_dir='', trans='resize'):
        self.root = root
        self.list_path = list_path
        self.crop_size = crop_size
        self.scale = scale
        self.ignore_label = ignore_label
        self.mean = mean
        self.is_mirror = mirror
        self.ssl_dir = ssl_dir
        self.img_ids = [i_id.strip() for i_id in open(list_path)]

        if not max_iters==None:
            self.img_ids = self.img_ids * int(np.ceil(float(max_iters) / len(self.img_ids)))
        self.files = []
        self.set = set
        self.trans = trans
        for name in self.img_ids:
            img_file = osp.join(self.root, "leftImg8bit/%s/%s" % (self.set, name))
            self.files.append({
                "img": img_file,
                "name": name
            })

    def __len__(self):
        return len(self.files)

    def __getitem__(self, index):
        datafiles = self.files[index]

        image = Image.open(datafiles["img"]).convert('RGB')
        name = datafiles["name"]

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

        if len(self.ssl_dir)>0:
            label = Image.open(osp.join(self.ssl_dir, name.replace('.png', '_labelTrainIds.png')))
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

class densepass13TestDataSet(data.Dataset):
    def __init__(self, root, list_path, max_iters=None, crop_size=(2048, 400), mean=(128, 128, 128),
                scale=False, mirror=False, ignore_label=255, set='val'):
        self.root = root
        self.list_path = list_path
        self.crop_size = crop_size
        self.scale = scale
        self.ignore_label = ignore_label
        self.mean = mean
        self.is_mirror = mirror
        self.set = set
        self.img_ids = [i_id.strip() for i_id in open(list_path)]
        if not max_iters==None:
            self.img_ids = self.img_ids * int(np.ceil(float(max_iters) / len(self.img_ids)))
        self.files = []

        for name in self.img_ids:
            img_file = osp.join(self.root, "leftImg8bit/%s/%s" % (self.set, name))
            lbname = name.replace(".png", "_labelTrainIds.png")
            label_file = osp.join(self.root, "gtFine/%s/%s" % (self.set, lbname))
            self.files.append({
                "img": img_file,
                "label": label_file,
                "name": name
            })
        self._key = np.array([0,1,2,3,4,5,6,7,8,9,10,11,11,12,12,12,255,12,12])

    def __len__(self):
        return len(self.files)

    def _map19to13(self, mask):
        values = np.unique(mask)
        new_mask = np.ones_like(mask) * 255
        # new_mask -= 1
        for value in values:
            if value == 255: 
                new_mask[mask==value] = 255
            else:
                new_mask[mask==value] = self._key[value]
        mask = new_mask
        return mask

    def __getitem__(self, index):
        datafiles = self.files[index]

        image = Image.open(datafiles["img"]).convert('RGB')
        label = Image.open(datafiles["label"])
        label = self._map19to13(np.array(label).astype('int32'))
        label = Image.fromarray(label)
        name = datafiles["name"]
        # resize
        image = image.resize(self.crop_size, Image.BICUBIC)
        label = label.resize(self.crop_size, Image.NEAREST)
        size = np.asarray(image).shape
        #
        input_transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((.485, .456, .406), (.229, .224, .225)),
        ])
        image = input_transform(image)

        label = torch.LongTensor(np.array(label).astype('int32'))
        return image, label, np.array(size), name


if __name__ == '__main__':
    dst = densepassTestDataSet("data/DensePASS_train_pseudo_val", 'dataset/densepass_list/val.txt', mean=(0,0,0))
    trainloader = data.DataLoader(dst, batch_size=4)
    for i, data in enumerate(trainloader):
        imgs, labels, *args = data
        if i == 0:
            img = torchvision.utils.make_grid(imgs).numpy()
            img = np.transpose(img, (1, 2, 0))
            img = img[:, :, ::-1]
            img = Image.fromarray(np.uint8(img) )
            img.show()
        break
