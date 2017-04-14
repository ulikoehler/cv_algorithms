#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Thinning algorithms
"""
import numpy as np
from ._ffi import *
from ._checks import *

__all__ = ["binary_directions"]

_ffi.cdef('''
int binary_directions(uint8_t* dst, const uint8_t* src, int width, int height);
''')

def binary_directions(img):
    """
    Takes a binary image and, for each pixel, computes
    which surrounding pixels are non-zero.
    Depdending on those pixels, bits in the uint8 output
    array are set or unset

    Parameters
    ==========
    img : numpy array-like
        A grayscale image that is assumed to be binary
        (every non-zero value is interpreted as 0).
        Usually this is a pre-thinned image.

    Returns
    =======
    A uint8-type output array the same shape of img,
    where the following bits are set or unset,
    if the respective neighbouring pixel is set or unset.

    Bit index by position:
    
        0 1 2
        3   4
        5 6 7

    Note that for Numpy due to the coordinate system,
    the respective pixels can be accessed like this:

        [y-1,x-1]  [y-1,x]  [y-1,x+1]
        [y,x-1]    [y,x]    [y,x+1]
        [y+1,x-1]  [y+1,x]  [y+1,x+1]

    The positions in this matrix correspond to the bit number
    shown above, e.g. bit #4 is (1 << 4) ORed to the result.
    """
    # Check if image has the correct type
    __check_image_grayscale_2d(img)
    img = force_c_order_contiguous(img)
    __check_array_uint8(img)

    height, width = img.shape

    # Allocate output array
    # uint32 is used so there is no overflow for large inputs
    out = np.zeros(img.shape, dtype=np.uint8, order="C")
    assert not np.isfortran(out)

    # Extract pointer to binary data
    srcptr = _ffi.cast("uint8_t*", img.ctypes.data)
    dstptr = _ffi.cast("uint8_t*", out.ctypes.data)

    rc = _libcv_algorithms.binary_directions(dstptr, srcptr, width, height)
    if rc != 0:
        raise ValueError("Internal error (return code {0}) in algorithm C code".format(rc))
    return out

    
