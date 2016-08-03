#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Utilities for image classification
"""
import numpy as np

__all__ = ["fractionWhite", "fractionBlack"]

def fractionWhite(img, minval=255, weights=None):
    """
    Given a binary image, counts the fraction of the pixels which are white,
    i.e. have a value above a specific threshold.

    Parameters
    ----------
    img : numpy (x,y) array
        The image to check
    minval : number
        Which value is the minimum to be considered white.
        The type must be the same as the image pixel type (usually int 0-255)
    weights : numpy (x,y) array
        An optional weights array (default 1) that modifies the
        weight of each pixel, if it passes the whiteness check
    """
    if len(img.shape) > 2:
        raise ValueError("Can only work with binary grayscale images")
    vals = img >= minval
    if weights is not None:
        vals = weights[vals]
    return np.sum(vals) / float(img.size)

def fractionBlack(img, maxval=0, weights=None):
    """
    Given a binary image, counts the fraction of the pixels which are black,
    i.e. have a value below a specific threshold
    
    Parameters
    ----------
    img : numpy (x,y) array
        The image to check
    maxval : number
        Which value is the maximum to be considered black.
        The type must be the same as the image pixel type (usually int 0-255)
    weights : numpy (x,y) array
        An optional weights array (default 1) that modifies the
        weight of each pixel, if it passes the blackness check
    """
    if len(img.shape) > 2:
        raise ValueError("Can only work with binary grayscale images")
    vals = img <= maxval
    if weights is not None:
        vals = weights[vals]
    return np.sum(vals) / float(img.size)
