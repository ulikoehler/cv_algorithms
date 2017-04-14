#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Thinning algorithms
"""
import numpy as np
from ._ffi import *
from ._checks import *

__all__ = ["grassfire"]

_ffi.cdef('''
int grassfire(uint32_t* dst, const uint8_t* mask, int width, int height);
''')

def grassfire(img):
    """
    Perform a grassfire transform on the given binary image.

    Parameters
    ==========
    img : numpy array-like
        A grayscale image that is assumed to be binary
        (every non-zero value is interpreted as 0).
        For example a countour image.

    Returns
    =======
    A uint32-type numpy array of the same dimensions
    as img, representing the grassfire count.
    """
    # Check if image has the correct type
    __check_image_grayscale_2d(img)
    img = force_c_order_contiguous(img)
    __check_array_uint8(img)

    height, width = img.shape

    # Allocate output array
    # uint32 is used so there is no overflow for large inputs
    out = np.zeros(img.shape, dtype=np.uint32, order="C")
    assert not np.isfortran(out)

    # Extract pointer to binary data
    maskptr = _ffi.cast("uint8_t*", img.ctypes.data)
    outptr = _ffi.cast("uint32_t*", out.ctypes.data)

    rc = _libcv_algorithms.grassfire(outptr, maskptr, width, height)
    if rc != 0:
        raise ValueError("Internal error (return code {0}) in algorithm C code".format(rc))
    return out

    
