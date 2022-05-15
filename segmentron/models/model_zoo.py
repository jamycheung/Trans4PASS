import logging
import torch

from collections import OrderedDict
from segmentron.utils.registry import Registry
from ..config import cfg

MODEL_REGISTRY = Registry("MODEL")
MODEL_REGISTRY.__doc__ = """
Registry for segment model, i.e. the whole model.

The registered object will be called with `obj()`
and expected to return a `nn.Module` object.
"""


def get_segmentation_model():
    """
    Built the whole model, defined by `cfg.MODEL.META_ARCHITECTURE`.
    """
    model_name = cfg.MODEL.MODEL_NAME
    model = MODEL_REGISTRY.get(model_name)()
    load_model_pretrain(model)
    return model


def load_model_pretrain(model):
    if cfg.PHASE == 'train':
        if cfg.TRAIN.PRETRAINED_MODEL_PATH:
            logging.info('load pretrained model from {}'.format(cfg.TRAIN.PRETRAINED_MODEL_PATH))
            state_dict_to_load = torch.load(cfg.TRAIN.PRETRAINED_MODEL_PATH)
            keys_wrong_shape = []
            state_dict_suitable = OrderedDict()
            state_dict = model.state_dict()
            for k, v in state_dict_to_load.items():
                if v.shape == state_dict[k].shape:
                    state_dict_suitable[k] = v
                else:
                    keys_wrong_shape.append(k)
            logging.info('Shape unmatched weights: {}'.format(keys_wrong_shape))
            msg = model.load_state_dict(state_dict_suitable, strict=False)
            logging.info(msg)
    else:
        if cfg.TEST.TEST_MODEL_PATH:
            logging.info('load test model from {}'.format(cfg.TEST.TEST_MODEL_PATH))
            model_dic = torch.load(cfg.TEST.TEST_MODEL_PATH, map_location='cuda:0')
            if 'state_dict' in model_dic.keys():
                model_dic = model_dic['state_dict']
            # --- load checkpoint from mmseg framework
            state_dict_suitable = OrderedDict()
            for k, v in model_dic.items():
                if 'backbone' in k:
                    state_dict_suitable[k.replace('backbone', 'encoder')] = v
                elif 'decode_head' in k:
                    state_dict_suitable[k.replace('decode_head', 'fpt_head')] = v
                else:
                    state_dict_suitable[k] = v
            msg = model.load_state_dict(state_dict_suitable, strict=False)
            logging.info(msg)