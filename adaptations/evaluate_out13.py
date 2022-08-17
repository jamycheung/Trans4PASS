import argparse
import scipy
from scipy import ndimage
import numpy as np
import sys
from packaging import version

import os, sys
os.chdir(sys.path[0])
import torch
from torch.autograd import Variable
import torchvision.models as models
import torch.nn.functional as F
from torch.utils import data, model_zoo
from model.trans4passplus import Trans4PASS_plus_v1, Trans4PASS_plus_v2
from dataset.dp13_dataset import densepass13DataSet, densepass13TestDataSet
# from dataset.cs_dataset_src import CSSrcDataSet
from torchvision import transforms
from compute_iou import fast_hist, per_class_iu

from collections import OrderedDict
from PIL import Image

import matplotlib.pyplot as plt
import torch.nn as nn
IMG_MEAN = np.array((104.00698793,116.66876762,122.67891434), dtype=np.float32)
SOURCE_NAME = 'CS13'
# SOURCE_NAME = 'SP13'
TARGET_NAME = 'DP13'
# MODEL = 'Trans4PASS_plus_v1'
MODEL = 'Trans4PASS_plus_v2'
EMB_CHANS = 128
DIR_NAME = '{}2{}_{}_MPA/'.format(SOURCE_NAME, TARGET_NAME, MODEL)
DATA_DIRECTORY = 'datasets/DensePASS'
DATA_LIST_PATH = 'dataset/densepass_list/val.txt'
INPUT_SIZE_TARGET = '2048,400'
SAVE_PATH = './result/' + DIR_NAME

IGNORE_LABEL = 255
NUM_CLASSES = 13
RESTORE_FROM = 'snapshots/CS132CS132DP13_Trans4PASS_plus_v2_MPA/BestCS132DP13_G.pth'
SET = 'val'

EMB_CHANS = 128
palette = [128, 64, 128, 
244, 35, 232, 
70, 70, 70, 
102, 102, 156, 
190, 153, 153, 
153, 153, 153, 
250, 170, 30,
220, 220, 0, 
107, 142, 35, 
152, 251, 152, 
70, 130, 180, 
220, 20, 60, 
# 255, 0, 0, 
0, 0, 142, 
# 0, 0, 70,
# 0, 60, 100, 
# 0, 80, 100, 
# 0, 0, 230, 
# 119, 11, 32
]
zero_pad = 256 * 3 - len(palette)


NAME_CLASSES = [
'road', 'sidewalk', 'building', 'wall', 'fence', 'pole', 'traffic light',
                'traffic sign', 'vegetation', 'terrain', 'sky', 'person', 'car']

for i in range(zero_pad):
    palette.append(0)


def colorize_mask(mask):
    # mask: numpy array of the mask
    new_mask = Image.fromarray(mask.astype(np.uint8)).convert('P')
    new_mask.putpalette(palette)

    return new_mask

def get_arguments():
    parser = argparse.ArgumentParser(description="Network")
    parser.add_argument("--model", type=str, default=MODEL,
                        help="available options : Trans4PASS_plus_v1, Trans4PASS_plus_v2")
    parser.add_argument("--emb-chans", type=int, default=EMB_CHANS,
                        help="Number of channels in decoder head.")
    parser.add_argument("--data-dir", type=str, default=DATA_DIRECTORY,
                        help="Path to the directory containing the Cityscapes dataset.")
    parser.add_argument("--data-list", type=str, default=DATA_LIST_PATH,
                        help="Path to the file listing the images in the dataset.")
    parser.add_argument("--ignore-label", type=int, default=IGNORE_LABEL,
                        help="The index of the label to ignore during the training.")
    parser.add_argument("--num-classes", type=int, default=NUM_CLASSES,
                        help="Number of classes to predict (including background).")
    parser.add_argument("--input-size-target", type=str, default=INPUT_SIZE_TARGET,
                        help="Comma-separated string with height and width of target images.")
    parser.add_argument("--restore-from", type=str, default=RESTORE_FROM,
                        help="Where restore model parameters from.")
    parser.add_argument("--gpu", type=int, default=0,
                        help="choose gpu device.")
    parser.add_argument("--set", type=str, default=SET,
                        help="choose evaluation set.")
    parser.add_argument("--save", type=str, default=SAVE_PATH,
                        help="Path to save result.")
    return parser.parse_args()


def main():
    args = get_arguments()

    gpu0 = args.gpu

    if not os.path.exists(args.save):
        os.makedirs(args.save)

    if args.model == 'Trans4PASS_plus_v1':
        model = Trans4PASS_plus_v1(num_classes=args.num_classes, emb_chans=args.emb_chans)
    elif args.model == 'Trans4PASS_plus_v2':
        model = Trans4PASS_plus_v2(num_classes=args.num_classes, emb_chans=args.emb_chans)
    else:
        raise ValueError

    saved_state_dict = torch.load(args.restore_from, map_location='cuda:0')
    if 'state_dict' in saved_state_dict.keys():
        saved_state_dict = saved_state_dict['state_dict']
    msg = model.load_state_dict(saved_state_dict, strict=False)
    print(msg)

    model.eval()
    model.cuda(gpu0)

    w, h = map(int, args.input_size_target.split(','))
    targettestset = densepass13TestDataSet(args.data_dir, args.data_list, crop_size=(w, h),
                                         mean=IMG_MEAN,
                                         scale=False, mirror=False, set='val')
    testloader = data.DataLoader(targettestset, batch_size=1, shuffle=False, pin_memory=True)
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((.485, .456, .406), (.229, .224, .225)),
    ])
    hist = np.zeros((args.num_classes, args.num_classes))
    interp = nn.Upsample(size=(h, w), mode='bilinear', align_corners=True)
    for index, batch in enumerate(testloader):
        if index % 100 == 0:
            print ('%d processd' % index)
        image, label, _, name = batch
        image = image.cuda(gpu0)
        b, _, _, _ = image.shape
        output_temp = torch.zeros((b, NUM_CLASSES, h, w), dtype=image.dtype).cuda(gpu0)
        scales = [1] #[0.5,0.75,1.0,1.25,1.5,1.75] # ms
        # scales = [0.75,1.0,1.75] # ms
        # scales = [0.5,0.75,1.0,1.25,1.5,1.75] # ms
        for sc in scales:
            new_h, new_w = int(sc * h), int(sc * w)
            img_tem = nn.UpsamplingBilinear2d(size=(new_h, new_w))(image)
            with torch.no_grad():
                _, output = model(img_tem)
                output_temp += interp(output)
        output = output_temp / len(scales)
        output = torch.argmax(output, 1).squeeze(0).cpu().data.numpy()

        label = label.cpu().data[0].numpy()
        hist += fast_hist(label.flatten(), output.flatten(), args.num_classes)

        save_vis = False
        if save_vis:
            output_col = colorize_mask(output)
            output = Image.fromarray(output.astype(np.uint8))
            name = name[0].split('/')[-1]
            output.save('%s/%s' % (args.save, name))
            output_col.save('%s/%s_color.png' % (args.save, name.split('.')[0]))


    mIoUs = per_class_iu(hist)
    for ind_class in range(args.num_classes):
        print('===>{:<15}:\t{}'.format(NAME_CLASSES[ind_class], str(round(mIoUs[ind_class] * 100, 2))))
    bestIoU = round(np.nanmean(mIoUs) * 100, 2)
    print('===> mIoU: ' + str(bestIoU))


if __name__ == '__main__':
    main()
