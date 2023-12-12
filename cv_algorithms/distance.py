#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Various distance metrics beteen images and related
data structures
"""
# -*- coding: utf-8 -*-
from ._ffi import *
import numpy as np

__all__ = ["pairwise_diff", "rgb_distance", "grayscale_distance"]

_ffi.cdef('''
int pairwise_diff(const double* a, const double* b, double* result, size_t awidth, size_t bwidth);
''')

def pairwise_diff(a, b):
    """
    Compute the pairwise |a-b| absolute difference between two 1D float arrays, a and b
    Returns a (a.size, b.size) float distance matrix.

    Uses a C-optimized backend.
    """
    if len(a.shape) != 1 or len(b.shape) != 1:
        raise ValueError("The arrays must have a 1D shape. Actual shape: {0} and {1}".format(a.shape, b.shape))
    if (a.nbytes // a.size) != 8 or (b.nbytes // b.size) != 8:
        raise ValueError("The arrays need to have 64-bit float elements")
    #  Allocate output array
    out = np.zeros((a.shape[0], b.shape[0]), np.float64)

    aptr = _ffi.cast("const double*", a.ctypes.data)
    bptr = _ffi.cast("const double*", b.ctypes.data)
    outptr = _ffi.cast("double*", out.ctypes.data)

    _libcv_algorithms.pairwise_diff(aptr, bptr, outptr, a.shape[0], b.shape[0])

    return out

def rgb_distance(img, color):
    """
    Compute the euclidean distance between
    the given color and each pixel in the image
    in the RGB space.

    Computes
    sqrt(rdelta² + gdelta² + bdelta²)

    Parameters
    ==========
    img : 2D RGB image as numpy array
        The RGB or BGR OpenCV image
    color : RGB 3-tuple

    Returns
    =======
    A numpy float array the same size as img,
    representing the pixel-to-color distances
    """
    imgfloat = img.astype(float)
    return np.sqrt(np.sum(np.square(imgfloat[:,:] - color), axis=2))

def grayscale_distance(img, value):
    """
    Compare a value.
    This is similar to

    np.abs(img - value)
    but it handles negative values in uint8 images correctly.

    Parameters
    ==========
    img
        A 2D numpy array
    value:
        Any grayscale value (single number) to compare to.

    Returns
    =======
    A floating point absolute difference image
    """
    return np.abs(img.astype(float) - float(value))
