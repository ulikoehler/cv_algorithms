#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Generic utilities
"""
import numpy as np
import math

__all__ = ["spread_to_grayscale"]

def spread_to_grayscale(img, spread_min=True):
    """
    Spreads the minimum/maximum range in the given
    grayscale image to (0-255) in the returned uint8 imge.

    This is intended to be used as an utility to export
    images for functions such as grassfire that return
    a datatype or value range that is not suitable for
    exporting or displaying a grayscale image.

    Parameters
    ==========
    img : numpy array-like
        The input image. Must be grayscale.
    spread_min : bool
        If True, the minimum value of the img is mapped
        to 0 in the result.

    Returns
    =======
    A uint8-dtyped image that returns
    """
    fimg = img.astype(float)
    fimg -= np.min(fimg) if spread_min else 0
    fmax = np.max(fimg)

    if math.fabs(fmax) < 1e-9: # if "fmax == 0"
        # Avoid divide by zero
        return fimg.astype(np.uint8)

    return (fimg * 255. / fmax).astype(np.uint8)
