import os
import os.path as osp
import numpy as np
import random
import matplotlib.pyplot as plt
import collections
import torch
import torchvision
from torch.utils import data
from PIL import Image
from torchvision import transforms


class CS13SrcDataSet(data.Dataset):
    def __init__(self, root, list_path, max_iters=None, crop_size=(321, 321), mean=(128, 128, 128), 
                scale=True, mirror=True, ignore_label=255, set='val'):
        self.root = root
        self.list_path = list_path
        self.crop_size = crop_size
        self.set = set
        self.img_ids = [i_id.strip() for i_id in open(list_path)]
        if not max_iters==None:
            self.img_ids = self.img_ids * int(np.ceil(float(max_iters) / len(self.img_ids)))
        self.files = []

        for name in self.img_ids:
            img_file = osp.join(self.root, "leftImg8bit/%s/%s" % (self.set, name))
            lbname = name.replace("leftImg8bit", "gtFine_labelTrainIds")
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
        size = np.array(image).shape
        input_transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((.485, .456, .406), (.229, .224, .225)),
        ])
        image = input_transform(image)

        label = torch.LongTensor(np.array(label).astype('int32'))
        return image, label, np.array(size), name


if __name__ == '__main__':
    dst = CSSrcDataSet("./data", is_transform=True)
    trainloader = data.DataLoader(dst, batch_size=4)
    for i, data in enumerate(trainloader):
        imgs, labels = data
        if i == 0:
            img = torchvision.utils.make_grid(imgs).numpy()
            img = np.transpose(img, (1, 2, 0))
            img = img[:, :, ::-1]
            plt.imshow(img)
            plt.show()
