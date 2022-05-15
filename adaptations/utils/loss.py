import torch
import torch.nn as nn
import torch.nn.functional as F

EPSILON = 1e-5

def transpose(x):
    return x.transpose(-2, -1)


def inverse(x):
    return torch.inverse(x)


def log_trace(x):
    x = torch.cholesky(x)
    diag = torch.diagonal(x, dim1=-2, dim2=-1)
    return 2 * torch.sum(torch.log(diag + 1e-8), dim=-1)


def log_det(x):
    return torch.logdet(x)

def kd_loss(feats, select_feat_self):
    T = 100
    alpha = 0.9
    loss_kl_self = nn.KLDivLoss()(
        F.log_softmax(feats / T, dim=1),
        F.softmax(select_feat_self / T, dim=1)) * (alpha * T * T) + F.cross_entropy(feats, torch.argmax(select_feat_self, dim=1).long()) * (1. - alpha)
    return loss_kl_self


def feat_kl_loss(feats, labels, feats_mem):
    B, C, H, W = feats.shape
    CLS = 19
    _, H_org, W_org = labels.shape
    labels = F.interpolate(labels.unsqueeze(1).float(), (H, W), mode='nearest')

    select_feat = torch.clone(feats)
    feats = feats.permute(0, 2, 3, 1).contiguous().view(-1, C)
    select_feat = select_feat.permute(0, 2, 3, 1).contiguous().view(-1, C)
    labels = labels.view(-1)

    feats_mem = feats_mem.squeeze(1)
    batch_feats_mem = torch.zeros_like(feats_mem)

    ignore_index = 255
    for c in labels.unique():
        if c == ignore_index: continue
        c = c.item()
        feats_cls = feats[labels == c].mean(0)
        batch_feats_mem[int(c)] = feats_cls

        m = labels == c
        m = m[..., None].repeat(1, C)
        feat_temp = feats_mem[int(c)][None, ...].expand(labels.shape[0], -1)
        select_feat = torch.where(m, feat_temp, select_feat)
    feats = feats.view(B, H, W, C).permute(0, 3, 1, 2).contiguous()
    select_feat = select_feat.view(B, H, W, C).permute(0, 3, 1, 2).contiguous()

    T = 20
    alpha = 0.9
    loss_kl = nn.KLDivLoss()(
        F.log_softmax(feats / T, dim=1),
        F.softmax(select_feat / T, dim=1)) * (alpha * T * T) + F.cross_entropy(feats, torch.argmax(select_feat, dim=1).long()) * (1. - alpha)


    return loss_kl, batch_feats_mem, select_feat
