#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Thinning algorithms
"""
import numpy as np
from ._ffi import *
from ._checks import *

__all__ = ["popcount"]

_ffi.cdef('''
int popcount8(uint8_t* dst, const uint8_t* src, int size);
int popcount16(uint8_t* dst, const uint16_t* src, int size);
int popcount32(uint8_t* dst, const uint32_t* src, int size);
int popcount64(uint8_t* dst, const uint64_t* src, int size);
''')

def popcount(arr):
    """
    Provides a population count implementation.
    The population count is the number of one bits.
    Based on GCC's __builtin_popcount().

    The implementation is written in C.

    Parameters
    ==========
    arr : numpy array
        Must have dtype of uint8, uint16, uint32
        or uint64.

    Returns
    =======
    A uint8 numpy array the same shape as array
    containing the pop
    """
    # Check if image has the correct type
    arr = force_c_order_contiguous(arr)

    # Allocate output array
    # uint32 is used so there is no overflow for large inputs
    out = np.zeros(arr.shape, dtype=np.uint8, order="C")
    assert not np.isfortran(out)

    # Extract pointer to binary data
    dstptr = _ffi.cast("uint8_t*", out.ctypes.data)

    if arr.dtype == np.uint8:
        fn = _libcv_algorithms.popcount8
        srcptr = _ffi.cast("uint8_t*", arr.ctypes.data)
    elif arr.dtype == np.uint16:
        fn = _libcv_algorithms.popcount16
        srcptr = _ffi.cast("uint16_t*", arr.ctypes.data)
    elif arr.dtype == np.uint32:
        fn = _libcv_algorithms.popcount32
        srcptr = _ffi.cast("uint32_t*", arr.ctypes.data)
    elif arr.dtype == np.uint64:
        fn = _libcv_algorithms.popcount64
        srcptr = _ffi.cast("uint64_t*", arr.ctypes.data)
    else:
        raise ValueError("popcount can only work on uint8, uint16, uint32 or uint64 array")

    rc = fn(dstptr, srcptr, arr.size)
    if rc != 0:
        raise ValueError("Internal error (return code {0}) in algorithm C code".format(rc))
    return out

    
