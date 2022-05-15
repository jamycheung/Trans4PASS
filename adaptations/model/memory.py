'''
Ref: https://github.com/CharlesPikachu/mcibi
'''
import numpy as np
import torch
import torch.distributed as dist
import torch.nn as nn
import torch.nn.functional as F
from dataset.cs_dataset_src import CSSrcDataSet
from sklearn.cluster import _kmeans
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.preprocessing import StandardScaler
from tqdm import tqdm
from torch.utils import data
from model.trans4pass import Trans4PASS_v1, Trans4PASS_v2
from dataset.densepass_dataset import densepassDataSet, densepassTestDataSet
import time

from collections import OrderedDict
NAME_CLASSES = [
    "road",
    "sidewalk",
    "building",
    "wall",
    "fence",
    "pole",
    "light",
    "sign",
    "vegetation",
    "terrain",
    "sky",
    "person",
    "rider",
    "car",
    "truck",
    "bus",
    "train",
    "motocycle",
    "bicycle"]

palette = [128, 64, 128, 244, 35, 232, 70, 70, 70, 102, 102, 156, 190, 153, 153, 153, 153, 153, 250, 170, 30,
           220, 220, 0, 107, 142, 35, 152, 251, 152, 70, 130, 180, 220, 20, 60, 255, 0, 0, 0, 0, 142, 0, 0, 70,
           0, 60, 100, 0, 80, 100, 0, 0, 230, 119, 11, 32]
palette = np.split(np.array(palette), 19)


'''define the memory cfg'''
memory_cfg = {
    'num_feats_per_cls': 1,
    'feats_len': 128,
    'ignore_index': 255,
    'align_corners': False,
    'savepath': 'init_memory_joint_ms.npy',
    'type': ['random_select', 'clustering'][1],
}

'''cluster by using cosine similarity'''
def cluster(sparse_data, nclust=1):
    def euc_dist(X, Y=None, Y_norm_squared=None, squared=False):
        return cosine_similarity(X, Y)
    _kmeans.euclidean_distances = euc_dist
    out = _kmeans.KMeans(n_clusters=nclust).fit(sparse_data)
    return out.cluster_centers_


def init_memory(dataloader_src, dataloader_trg, backbone_net, num_classes=19, save_path=None):
    backbone_net = backbone_net.cuda()
    backbone_net.eval()
    assert memory_cfg['type'] in ['random_select', 'clustering']
    # extract feats
    FloatTensor = torch.cuda.FloatTensor
    feats_dict_s = {}
    feats_dict_s_len = []
    memory_s = np.zeros((num_classes, memory_cfg['num_feats_per_cls'], memory_cfg['feats_len']))

    feats_dict_t = {}
    feats_dict_t_len = []
    memory_t = np.zeros((num_classes, memory_cfg['num_feats_per_cls'], memory_cfg['feats_len']))
    # interp = nn.Upsample(size=(400, 2048), mode='bilinear', align_corners=True)

    for i in range(2):
        dataloader = [dataloader_src, dataloader_trg][i]
        feats_dict = [feats_dict_s, feats_dict_t][i]
        memory = [memory_s, memory_t][i]
        name = ['src', 'trg'][i]
        path_mem = memory_cfg['savepath'].replace('.npy', '_{}.npy'.format(name)) if save_path is None else save_path
        path_feats_dict = memory_cfg['savepath'].replace('.npy', '_{}_feats_dict.npy'.format(name)) if save_path is None else save_path.replace('.npy', '_feats_dict.npy')

        pbar = tqdm(enumerate(dataloader))
        print('Init memory.')
        for batch_idx, samples in pbar:
            if batch_idx % 100 ==0:
                pbar.set_description('Processing %s/%s...' % (batch_idx+1, len(dataloader)))
            image, gt, _, _ = samples
            image = image.type(FloatTensor)
            gt = gt.to(image.device)
            b, c, h, w = image.shape
            pred_temp = torch.zeros((b, num_classes, h, w), dtype=image.dtype).to(image.device)
            feats_temp = torch.zeros((b, memory_cfg['feats_len'], h//4, w//4), dtype=image.dtype).to(image.device)
            scales = [1] #[0.5, 0.75, 1.0, 1.25, 1.5, 1.75] for multi scales
            for sc in scales:
                new_h, new_w = int(sc * h), int(sc * w)
                img_tem = nn.UpsamplingBilinear2d(size=(new_h, new_w))(image)
                with torch.no_grad():
                    feats, pred = backbone_net(img_tem)
                    feat_fused = sum(feats)
                    pred_temp += nn.UpsamplingBilinear2d(size=(h, w))(pred)
                    feats_temp += nn.UpsamplingBilinear2d(size=(h//4, w//4))(feat_fused)
            pred = pred_temp / len(scales)
            feat_fused = feats_temp / len(scales)
            pred_prob = torch.softmax(pred, dim=1)
            conf, pred_cls = torch.max(pred_prob, dim=1) # pred_cls
            pred_cls = F.interpolate(pred_cls.unsqueeze(dim=1).float(), size=feat_fused.shape[-2:], mode='nearest')
            pred_cls = pred_cls.to(image.device)
            gt = F.interpolate(gt.unsqueeze(dim=1).float(), size=feat_fused.shape[-2:], mode='nearest')
            num_channels = feat_fused.size(1)
            clsids = gt.unique()
            feat_fused = feat_fused.permute(0, 2, 3, 1).contiguous() # B, H, W, C
            feat_fused = feat_fused.view(-1, num_channels) # BHW, C
            for clsid in clsids:
                clsid = int(clsid.item())
                if clsid == memory_cfg['ignore_index']: continue
                seg_cls = gt.view(-1) # BHW
                pred_cls = pred_cls.view(-1)
                correct_feat_mask = torch.logical_and(seg_cls==clsid, pred_cls==clsid)
                if torch.count_nonzero(correct_feat_mask) < 1:
                    continue
                feats_cls = feat_fused[correct_feat_mask].mean(0).data.cpu()
                if clsid in feats_dict:
                    feats_dict[clsid].append(feats_cls.unsqueeze(0).numpy())
                else:
                    feats_dict[clsid] = [feats_cls.unsqueeze(0).numpy()] # (19, N/Batch=num_samples, (1, feats_len))
        if i == 0:
            feats_dict_s_len = [len(feats_dict[j]) for j in range(num_classes)]
        else:
            feats_dict_t_len = [len(feats_dict[j]) for j in range(num_classes)]

        assert len(feats_dict) == num_classes
        for idx in range(num_classes):
            feats_cls_list = [torch.from_numpy(item) for item in feats_dict[idx]]
            feats_cls_list = torch.cat(feats_cls_list, dim=0).numpy()  # (N/Batch, feats_len) for each clsid/cluster_center
            memory[idx] = np.mean(feats_cls_list, axis=0)

    # --- joint
    feats_dict = {}
    for k, v in feats_dict_s.items():
        feats_dict[k] = feats_dict_s[k] + feats_dict_t[k]
    memory = np.zeros((num_classes, memory_cfg['num_feats_per_cls'], memory_cfg['feats_len']))
    for i in range(num_classes):
        memory[i] = (memory_s[i]+memory_t[i]) / 2.
    np.save(memory_cfg['savepath'], memory)

    return memory


if __name__ == '__main__':
    INPUT_SIZE = '1024,512'
    data_dir = 'datasets/Cityscapes'
    data_list = 'dataset/cityscapes_list/train.txt'
    trainset = CSSrcDataSet(data_dir, data_list, crop_size=(1024, 512), set='train')
    trainloader = data.DataLoader(trainset, batch_size=2, shuffle=False, num_workers=4, pin_memory=True)

    INPUT_SIZE_TARGET_TEST = '2048,400'
    data_dir_trg = 'datasets/DensePASS'
    data_list_trg = 'dataset/densepass_list/train.txt'
    SSL_DIR = './pseudo_DensePASS_Trans4PASS_v1_ms'
    targettestset = densepassDataSet(data_dir_trg, data_list_trg, crop_size=(2048, 400), set='train', ssl_dir=SSL_DIR)
    testloader = data.DataLoader(targettestset, batch_size=2, shuffle=False, num_workers=4, pin_memory=True)
    # --- warm up model
    RESTORE_FROM = 'snapshots/CS2DensePASS_Trans4PASS_v1_WarmUp/BestCS2DensePASS_G.pth'

    model = Trans4PASS_v1(num_classes=19, emb_chans=128)
    saved_state_dict = torch.load(RESTORE_FROM, map_location=lambda storage, loc: storage)
    if 'state_dict' in saved_state_dict.keys():
        saved_state_dict = saved_state_dict['state_dict']
    new_saved_state_dict = OrderedDict()
    for k, v in saved_state_dict.items():
        if 'backbone.' in k:
            new_saved_state_dict[k.replace('backbone.', 'encoder.')] = v
        elif 'decode_head.' in k:
            new_saved_state_dict[k.replace('decode_head.', 'dede_head.')] = v
        else:
            new_saved_state_dict[k] = v
    saved_state_dict = new_saved_state_dict
    msg = model.load_state_dict(saved_state_dict, strict=False)
    # print(msg)

    device = torch.device("cuda")
    model.to(device)
    model.eval()

    with torch.no_grad():
        init_memory(trainloader, testloader, model, 19)
