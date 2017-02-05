#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Utilities regarding text placement in images.
Supplements OpenCV's putText()
"""
import cv2
import numpy as np

__all__ = ["putTextCenter", "putTextAutoscale"]

def putTextCenter(img, txt, coords, fontFace, fontScale=1.,
                  color=(0,0,0), thickness=2, hshift=0., vshift=0., baseline_shift=0.):
    """
    Like cv2.putText(), but auto-centers text on the given coordinates
    by computing size via cv2.getTextSize() and shifting the coordinates by that amount.

    Additional horizontal and vertical shifts can be performed by using nonzero
    hshift/vshift parameters.

    The actual text center is used as a reference, not the text baseline.
    However, baseline_shift, multiplied by the baseline, can be added to
    the value. Setting baseline_shift=1. will result in one (baseline - bottom)
    difference being added to the y coordinate, resulting in the image being shifted up.

    Parameters
    ----------
    img : numpy ndarray
        The image that is passed to OpenCV
    txt : str
        The text to render
    coords : (int, int)
        The coordinates where the text center should be placed
    fontFace
        The fontFace parameter that is passed to OpenCV
    fontScale
        The fontScale parameter that is passed to OpenCV
    color
        The font color that will be rendered
    thickness
        The rendering thickness
    hshift : float
        Horizontal shift, will be added to the x coordinate of the render position
    vshift : float
        Horizontal shift, will be added to the x coordinate of the render position
    baseline_shift : float
        Factor of the (baseline - bottom) y difference that will be added to the
        y coordinate of the rendering position
    """
    (w,h), baseline = cv2.getTextSize(txt, fontFace, fontScale, thickness)
    coords = (int(round(coords[0] - w/2) + hshift),
              int(round(coords[1] + h/2) + vshift + baseline_shift * baseline))
    cv2.putText(img, txt, coords, fontFace, fontScale, color, thickness)


def putTextAutoscale(img, txt, coords, fontFace, w, h, heightFraction=0.5, widthFraction=0.95,
                     maxHeight=60, color=(0,0,0), thickness=2, hshift=0., vshift=0., baseline_shift=0.):
    """
    Like cv2.putText(), but auto-centers text on the given coordinates and automatically
    chooses the text size for a (w*h) box around the center so that it consumes heightFraction*h height.
    Also ensures that the text consumes no more than widthFraction*w space horizontally,
    but in any case the text will not consume more than maxHeight pixels vertically.
    
    This is done by computing size via cv2.getTextSize() and shifting and scaling appropriately.

    Parameters
    ----------
    img : numpy ndarray
        The image that is passed to OpenCV
    txt : str
        The text to render
    coords : (int, int)
        The coordinates where the text center should be placed
    fontFace
        The fontFace parameter that is passed to OpenCV
    w : float
        The width of the box to place the text into
    h : float
        The height of the box to place the text into
    heightFraction : float
        Represents the maximum fraction of the height of the box
        that will be occupied by the text box
    widthFraction : float
        Represents the maximum fraction of the width of the box
        that will be occupied by the text box
    maxHeight : int
        The maximum height of the text box in pixels
    color
        The font color that will be rendered
    thickness
        The rendering thickness
    hshift : float
        Horizontal shift, will be added to the x coordinate of the render position
    vshift : float
        Horizontal shift, will be added to the x coordinate of the render position
    baseline_shift : float
        Factor of the (baseline - bottom) y difference that will be added to the
        y coordinate of the rendering position
    """
    # Get first size estimate. We will scale fractionally.
    (scale1Width, scale1Height), baseline = cv2.getTextSize(txt, fontFace, 1., thickness)
    # What are the different maximum scales that are mandated by the width and height
    heightFractionScale = (heightFraction * h) / scale1Height
    widthFractionScale = (widthFraction * w) / scale1Width
    absMaxScale = maxHeight / scale1Height
    # Combine height scale, width scale and absolute max height
    newScale = min(heightFractionScale, widthFractionScale, absMaxScale)
    # Now place text using putTextCenter()
    putTextCenter(img, txt, coords, fontFace, newScale,
                    color, thickness, hshift, vshift, baseline_shift)
