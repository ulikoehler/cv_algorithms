#!/usr/bin/env python3
from ._ffi import *

_ffi.cdef('''
int xy_distance(const double* a, const double* b, double* result, size_t awidth, size_t bwidth, size_t height);
''')

def xy_distance(a, b):
    if len(a.shape) != 2 or len(b.shape) != 2:
        raise ValueError("The arrays must have a 2D shape. Actual shape: {0} and {1}".format(a.shape, b.shape))
    if a.shape[1] != b.shape[1]:
        raise ValueError("The 2nd dimensionshapes of the arrays to compare don't match: {0} and {1}".format(a.shape, b.shape))
    if (a.nbytes / a.size) != 8. or (b.nbytes / b.size) != 8.:
        raise ValueError("The given arrays need to have 64-bit float elements")
    #  Allocate output array
    out = np.zeros((a.shape[0], b.shape[0], a.shape[1]), np.float64)

    aptr = _ffi.cast("const double*", a.ctypes.data)
    bptr = _ffi.cast("const double*", b.ctypes.data)
    outptr = _ffi.cast("double*", out.ctypes.data)

    _libcv_algorithms.xy_distance(aptr, bptr, outptr, a.shape[0], b.shape[0], a.shape[1])

    return out