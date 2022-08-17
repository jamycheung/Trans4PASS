import torch
import json
import os
import os.path as osp
from torchvision import transforms
import glob
import numpy as np
import torchvision
from PIL import Image
from torch.utils import data


class Structured3D8DataSet(data.Dataset):
    def __init__(self, root, list_path, max_iters=None, crop_size=(321, 321), mean=(128, 128, 128),
                 scale=True, mirror=True, ignore_label=0, set='val', fold=None):
        self.root = root
        self.crop_size = crop_size

        self.img_ids = [i_id.strip() for i_id in open(list_path)]
        if not max_iters==None:
            self.img_ids = self.img_ids * int(np.ceil(float(max_iters) / len(self.img_ids)))

        self.files = []

        for name in self.img_ids:
            img_file = osp.join(self.root, name)
            label_file = img_file.replace("rgb_rawlight", "semantic")
            self.files.append({
                "img": img_file,
                "label": label_file,
                "name": name
            })
        self._key = np.array([255,6,3,255,255,1,4,5,2,7,255,255,255,255,255,255,255,255,255,255,255,255,0,255,255,255,255,255,255,255,255,255,255,255,255,255,255,255,255,255,255])

    def __len__(self):
        return len(self.files)

    def _map40to8(self, mask):
        values = np.unique(mask)
        for value in values:
            if value == 255: 
                mask[mask==value] = 255
            else:
                mask[mask==value] = self._key[value]
        return mask

    def __getitem__(self, index):
        datafiles = self.files[index]

        image = Image.open(datafiles["img"]).convert('RGB')
        label = Image.open(datafiles["label"])
        label = self._map40to8(np.array(label).astype('int32'))
        label = Image.fromarray(label)
        name = datafiles["name"]
        # resize
        image = image.resize(self.crop_size, Image.BICUBIC)
        label = label.resize(self.crop_size, Image.NEAREST)
        size = np.asarray(image).shape

        input_transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((.485, .456, .406), (.229, .224, .225)),
        ])
        image = input_transform(image)
        label = torch.LongTensor(np.array(label).astype('int32'))

        return image, label, np.array(size), name


    def _vis(self, rgb, sem):
        # Visualization
        vis = np.array(rgb)
        vis = vis // 2 + self.colors[sem] // 2
        Image.fromarray(vis).show()

if __name__ == '__main__':
    dst = Structured3D8DataSet("data/Structured3D", 'dataset/s2d3d_pin_list/train.txt', mean=(0,0,0))
    trainloader = data.DataLoader(dst, batch_size=4)
    for i, data in enumerate(trainloader):
        imgs, labels, size, name = data
        if i == 0:
            img = torchvision.utils.make_grid(imgs).numpy()
            img = np.transpose(img, (1, 2, 0))
            img = img[:, :, ::-1]
            img = Image.fromarray(np.uint8(img) )
            img.show()
        break
