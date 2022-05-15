import torch
import torch.nn as nn
import numpy as np
import torch.nn.functional as F


def global_avg_pool(inputs, weight):
    b,c,h,w = inputs.shape[-4], inputs.shape[-3], inputs.shape[-2], inputs.shape[-1]
    weight_new = weight.detach().clone()
    weight_sum = torch.sum(weight_new)
    weight_new = weight_new.view(h,w)
    weight_new = weight_new.expand(b,c,h,w)
    weight_sum = max(weight_sum, 1e-12)
    return torch.sum(inputs*weight_new,dim=(-1,-2),keepdim=True) / weight_sum

