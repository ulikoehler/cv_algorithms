#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Utilities for extracting color channels out of arbitrary images
"""
import cv2
import numpy as np
import enum

__all__ = ["ColorspaceChannel", "Colorspace", "convert_to_colorspace",
    "extract_channel"]


class Colorspace(enum.IntEnum):
    BGR = 0 # "Standard" OpenCV colorspace
    RGB = 1
    HSV = 2
    LAB = 3
    YUV = 4
    YCrCb = 5
    HLS = 6
    LUV = 7
    XYZ = 8

    @property
    def channels(colspace):
        """
        Get a tuple of all three ColorspaceChannels
        for the given colorspace
        """
        return (ColorspaceChannel(colspace.value * 3),
                ColorspaceChannel(colspace.value * 3 + 1),
                ColorspaceChannel(colspace.value * 3 + 2))

class ColorspaceChannel(enum.IntEnum):
    """
    Different types of color channels
    """
    # BGR
    BGR_Blue = 0
    BGR_Green = 1
    BGR_Red = 2
    # RGB
    RGB_Red = 3
    RGB_Green = 4
    RGB_Blue = 5
    # HSV
    HSV_Hue = 6
    HSV_Saturation = 7
    HSV_Value = 8
    # LAB
    LAB_L = 9
    LAB_a = 10
    LAB_b = 11
    # YUV
    YUV_Luma = 12 # Y
    YUV_U = 13
    YUV_V = 14
    # YCrCb
    YCrCb_Luma = 15 # Y
    YCrCb_Cr = 16
    YCrCb_Cb = 17
    # HLS
    HLS_Hue = 18
    HLS_Lightness = 19
    HLS_Saturation = 20
    # LUV
    LUV_L = 21
    LUV_U = 22
    LUV_V = 23
    # XYZ
    XYZ_X = 24
    XYZ_Y = 25
    XYZ_Z = 26
    
    
    @property
    def colorspace(self):
        """
        Get the colorspace for the current instance
        """
        return Colorspace(self.value // 3)
    
    @property
    def channel_idx(self):
        """
        Get the channel number for the colorspace (0 to 2)
        """
        return self.value % 3


# Arguments to convert BGR to another colorspace
_colorspace_cvt_from_bgr = {
    Colorspace.BGR: None,
    Colorspace.RGB: cv2.COLOR_BGR2RGB,
    Colorspace.HSV: cv2.COLOR_BGR2HSV,
    Colorspace.LAB: cv2.COLOR_BGR2LAB,
    Colorspace.YUV: cv2.COLOR_BGR2YUV,
    Colorspace.YCrCb: cv2.COLOR_BGR2YCrCb,
    Colorspace.HLS: cv2.COLOR_BGR2HLS,
    Colorspace.LUV: cv2.COLOR_BGR2LUV,
    Colorspace.XYZ: cv2.COLOR_BGR2XYZ
}

# Arguments to convert a colorspace to BGR
_colorspace_cvt_to_bgr = {
    Colorspace.BGR: None,
    Colorspace.RGB: cv2.COLOR_RGB2BGR,
    Colorspace.HSV: cv2.COLOR_HSV2BGR,
    Colorspace.LAB: cv2.COLOR_LAB2BGR,
    Colorspace.YUV: cv2.COLOR_YUV2BGR,
    Colorspace.YCrCb: cv2.COLOR_YCrCb2BGR,
    Colorspace.HLS: cv2.COLOR_HLS2BGR,
    Colorspace.LUV: cv2.COLOR_LUV2BGR,
    Colorspace.XYZ: cv2.COLOR_XYZ2BGR
}

def convert_to_colorspace(img, new_colorspace, source=Colorspace.BGR):
    """
    Convert an image in an arbitrary colorspace
    to another colorspace using OpenCV

    Parameters
    ==========
    img : NumPy image
        Any supported OpenCV image
    new_colorspace : Colorspace enum
        The target colorspace
    source : Colorspace enum
        The source colorspace.
        If in doubt, BGR is probably right

    Returns
    =======
    The converted image, or img if
    source == target.
    """
    # Convert from source to BGR
    if source != Colorspace.BGR:
        img = cv2.cvtColor(img, _colorspace_cvt_to_bgr[source])
    # Convert to target
    cvt = _colorspace_cvt_from_bgr[new_colorspace]
    if cvt is None: # Already in target
        return img
    return cv2.cvtColor(img, cvt)

def extract_channel(img, channel, source=Colorspace.BGR):
    """
    Extract a single channel from an arbitrary colorspace
    from an image

    Parameters
    ==========
    img : NumPy / OpenCV image
    channel : ColorspaceChannel enum
        The target channel
    source : Colorspace enum
        The current colorspace of the imge

    Returns
    =======
    The resulting channel as a NumPy image.
    The returned array is similar to a grayscale image.
    """
    target_space = channel.colorspace
    # Convert to the correct colorspace
    img = convert_to_colorspace(img, target_space, source)
    # Extract appropriate channel
    return img[:,:,channel.channel_idx]

