# Copyright (c) CAIRI AI Lab. All rights reserved
# Code adapted from:
# https://github.com/chengtan9907/OpenSTL

from .cotere import COTERE

method_maps = {
    'cotere': COTERE
}

__all__ = [
    'COTERE'
]