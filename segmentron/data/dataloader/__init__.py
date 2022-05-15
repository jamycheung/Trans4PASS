"""
This module provides data loaders and transformers for popular vision datasets.
"""
# from .mscoco import COCOSegmentation
from .cityscapes import CitySegmentation
from .stanford2d3d import Stanford2d3dSegmentation
from .stanford2d3d_pan import Stanford2d3dPanSegmentation
from .densepass import DensePASSSegmentation

datasets = {
    'cityscape': CitySegmentation,
    'stanford2d3d': Stanford2d3dSegmentation,
    'stanford2d3d_pan': Stanford2d3dPanSegmentation,
    'densepass': DensePASSSegmentation,
}


def get_segmentation_dataset(name, **kwargs):
    """Segmentation Datasets"""
    return datasets[name.lower()](**kwargs)
