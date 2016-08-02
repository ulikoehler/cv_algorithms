#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Contour utilities
"""
import cv2
import numpy as np

__all__ = ["meanCenter", "scaleByRefpoint"]


def meanCenter(contour):
    """
    Compute the center of a contour by taking the mean of all coordinates

    Parameters
    ----------
    contour
        (n,2) numpy array of coordinates
    """
    return np.mean(contour, axis=0)


def scaleByRefpoint(contour, xscale=1., yscale=1., refpoint=None):
    """
    Scale a contour around a reference point.
    Takes a (n,2) coordinate array and optionally a reference point.
    If the reference point is None, it is computed from the contour

    Parameters
    ----------
    contour
        A (n,2) array of 2D coordinates
    xscale
        The x axis scale factor
    yscale
        The y axis scale factor
    """
    if refpoint is None:
        refpoint = meanCenter(contour)
    # Create transformation matrix
    refx, refy = refpoint
    contour = contour.copy()  # Don't work on original
    # Shift so that refpoint is origin
    contour[:, 0] -= refx
    contour[:, 1] -= refy
    # Apply scale
    contour[:, 0] *= xscale
    contour[:, 1] *= yscale
    # Shift reference point
    contour[:, 0] += refx
    contour[:, 1] += refy
    return contour
