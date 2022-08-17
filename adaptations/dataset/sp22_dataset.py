import os.path as osp
import os.path as osp

import numpy as np
import torch
import torchvision
from PIL import Image
from torch.utils import data
from torchvision import transforms

from utils.transform import FixScaleRandomCropWH


class synpassDataSet(data.Dataset):
    def __init__(self, root, list_path, max_iters=None, crop_size=(321, 321), mean=(128, 128, 128), 
                scale=True, mirror=True, ignore_label=255, set='val'):
        self.root = root
        self.list_path = list_path
        self.crop_size = crop_size
        self.scale = scale
        self.ignore_label = ignore_label
        self.mean = mean
        self.is_mirror = mirror
        self.img_ids = [i_id.strip() for i_id in open(list_path)]
        if not max_iters==None:
            self.img_ids = self.img_ids * int(np.ceil(float(max_iters) / len(self.img_ids)))
        self.files = []
        self.set = set
        for name in self.img_ids:
            img_file = osp.join(self.root, name)
            lbname = name.replace("img", "semantic").replace('.jpg', '_trainID.png')
            label_file = osp.join(self.root, lbname)
            self.files.append({
                "img": img_file,
                "label": label_file,
                "name": name
            })
        self._key = np.array([255,2,4,255,11,5,0,0,1,8,12,3,7,10,255,255,255,255,6,255,255,255,9])

    def __len__(self):
        return len(self.files)

    def _map23to22(self, mask):
        mask[mask==0] = 255
        # values = np.unique(mask)
        # new_mask = np.ones_like(mask) * 255
        # # new_mask -= 1
        # for value in values:
        #     if value == 255: 
        #         new_mask[mask==value] = 255
        #     else:
        #         new_mask[mask==value] = self._key[value]
        # mask = new_mask
        return mask

    def __getitem__(self, index):
        datafiles = self.files[index]

        image = Image.open(datafiles["img"]).convert('RGB')
        label = Image.open(datafiles["label"])
        label = self._map23to22(np.array(label).astype('int32'))
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
    dst = synpassTestDataSet("data/SynPASS", 'dataset/SynPASS_list/val.txt', mean=(0,0,0))
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
