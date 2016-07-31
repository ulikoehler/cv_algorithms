#!/usr/bin/env python
# -*- coding: utf-8 -*-
from cffi import FFI
import imp
import os
import sys
import cv2

ffi = FFI()
ffi.cdef('''
int guo_hall_thinning(uint8_t* binary_image, size_t width, size_t height);
int zhang_suen_thinning(uint8_t* binary_image, size_t width, size_t height);
''')

# Open native library
if sys.version_info >= (3, 4):
    import importlib
    soname = importlib.util.find_spec("cv_algorithms._cv_algorithms").origin
else:
    curmodpath = sys.modules[__name__].__path__
    soname = imp.find_module('_cv_algorithms', curmodpath)[1]

__libcv_algorithms = ffi.dlopen(soname)

def guo_hall(img):
    """
    Guo-Hall variant that works on a copy of the image.
    See guo_hall_inplace() docs for more info.
    """
    return guo_hall_inplace(img.copy())

def guo_hall_inplace(img):
    """
    Perform in-place optimized Guo-Hall thinning.
    Returns img.

    Requires a binary grayscale numpy array as input.

    This calls the optimized C backend from cv_algorithms.
    """
    # Check if image seems correct
    nd = len(img.shape)
    if nd == 3:
        raise ValueError("Can only use binary (i.e. grayscale) images")
    if nd != 2:
        raise ValueError("img has wrong number of dimensions ({0} instead of 2)".format(nd))

    # Check order TODO

    # Can't perform Guo-hall on 0-2 pixel wide/high images
    height, width = img.shape
    if height < 3 or width < 3:
        raise ValueError("Guo-Hall algorithm needs an image at least 3px wide and 3px high")
    # Extract pointer to binary data
    dptr = ffi.cast("uint8_t*" , img.ctypes.data)

    rc = __libcv_algorithms.guo_hall_thinning(dptr, width, height)
    if rc != 0:
        raise ValueError("Internal error (return code {0}) in algorithm C code".format(rc))
    return img


def zhang_suen(img):
    """
    Zhang-Suen variant that works on a copy of the image.
    See zhang_suen_inplace() docs for more info.
    """
    return zhang_suen_inplace(img.copy())

def zhang_suen_inplace(img):
    """
    Perform in-place optimized Zhang-Suen thinning.
    Returns img.

    Requires a binary grayscale numpy array as input.

    This calls the optimized C backend from cv_algorithms.
    """
    # Check if image seems correct
    nd = len(img.shape)
    if nd == 3:
        raise ValueError("Can only use binary (i.e. grayscale) images")
    if nd != 2:
        raise ValueError("img has wrong number of dimensions ({0} instead of 2)".format(nd))

    # Check order TODO

    # Can't perform Guo-hall on 0-2 pixel wide/high images
    height, width = img.shape
    if height < 3 or width < 3:
        raise ValueError("Guo-Hall algorithm needs an image at least 3px wide and 3px high")
    # Extract pointer to binary data
    dptr = ffi.cast("uint8_t*" , img.ctypes.data)

    rc = __libcv_algorithms.zhang_suen_thinning(dptr, width, height)
    if rc != 0:
        raise ValueError("Internal error (return code {0}) in algorithm C code".format(rc))
    return img
