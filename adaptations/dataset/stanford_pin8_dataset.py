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

__FOLD__ = {
    '1_train': ['area_1', 'area_2', 'area_3', 'area_4', 'area_6'],
    '1_val': ['area_5a', 'area_5b'],
    '2_train': ['area_1', 'area_3', 'area_5a', 'area_5b', 'area_6'],
    '2_val': ['area_2', 'area_4'],
    '3_train': ['area_2', 'area_4', 'area_5a', 'area_5b'],
    '3_val': ['area_1', 'area_3', 'area_6'],
    'trainval': ['area_1', 'area_2', 'area_3', 'area_4', 'area_5a', 'area_5b', 'area_6'],
}

class StanfordPin8DataSet(data.Dataset):
    def __init__(self, root, list_path, max_iters=None, crop_size=(321, 321), mean=(128, 128, 128),
                 scale=True, mirror=True, ignore_label=0, set='val', fold=1):
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
        self.colors = np.load('dataset/s2d3d_pin_list/colors.npy')
        self.id2label = np.array([name2id[name] for name in id2name], np.uint8)
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

    def __getitem__(self, index):
        datafiles = self.files[index]

        image = Image.open(datafiles["img"]).convert('RGB')
        label = Image.open(datafiles["label"])
        label = self._color2id(image, label)
        label = self._map13to8(np.array(label).astype('int32'))
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

    def _color2id(self, img, sem):
        sem = np.array(sem, np.int32)
        rgb = np.array(img, np.int32)
        unk = (sem[..., 0] != 0)
        sem = self.id2label[sem[..., 1] * 256 + sem[..., 2]]
        sem[unk] = 0
        sem[rgb.sum(-1) == 0] = 0
        sem -= 1
        return Image.fromarray(sem)

    def _vis(self, rgb, sem):
        # Visualization
        vis = np.array(rgb)
        vis = vis // 2 + self.colors[sem] // 2
        Image.fromarray(vis).show()

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
        img_paths += glob.glob(os.path.join(folder, '{}/data/rgb/*_rgb.png'.format(a)))
    img_paths = sorted(img_paths)
    return img_paths

if __name__ == '__main__':
    dst = StanfordPinDataSet("data/Stanford2D3D", 'dataset/s2d3d_pin_list/train.txt', mean=(0,0,0))
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
