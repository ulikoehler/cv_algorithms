#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Thinning algorithms
"""
import numpy as np
from ._ffi import *
from ._checks import *

__all__ = ["guo_hall", "zhang_suen"]

_ffi.cdef('''
int guo_hall_thinning(uint8_t* binary_image, size_t width, size_t height);
int zhang_suen_thinning(uint8_t* binary_image, size_t width, size_t height);
''')


def __run_thinning(img, inplace, cfun):
    """Internal common thinning function"""
    # Copy image (it'll be changed by the C code) if not allowed to modify
    if not inplace:
        img = img.copy()
    # Check if image seems correct
    __check_image_grayscale_2d(img)
    img = force_c_order_contiguous(img)
    __check_image_min_wh(img, 3, 3)
    __check_array_uint8(img)

    height, width = img.shape

    # Extract pointer to binary data
    dptr = _ffi.cast("uint8_t*", img.ctypes.data)

    rc = cfun(dptr, width, height)
    if rc != 0:
        raise ValueError("Internal error (return code {0}) in algorithm C code".format(rc))
    return img


def guo_hall(img, inplace=False):
    """
    Perform in-place optimized Guo-Hall thinning.
    Returns img.

    Requires a binary grayscale numpy array as input.

    This calls the optimized C backend from cv_algorithms.
    """
    return __run_thinning(img, inplace, _libcv_algorithms.guo_hall_thinning)
    

def zhang_suen(img, inplace=False):
    """
    Perform in-place optimized Zhang-Suen thinning.
    Returns img.

    Requires a binary grayscale numpy array as input.

    This calls the optimized C backend from cv_algorithms.
    """
    
    return __run_thinning(img, inplace, _libcv_algorithms.zhang_suen_thinning)
