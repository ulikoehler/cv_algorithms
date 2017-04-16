#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Utilities for extracting color channels out of arbitrary images
"""
import cv2
import numpy as np
import enum

__all__ = ["ColorspaceChannel", "Colorspace"]


class Colorspace(enum.IntEnum):
    RGB = 0
    HSV = 1
    LAB = 2
    YUV = 3
    YCrCb = 4
    HLS = 5
    LUV = 6
    XYZ = 7

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
    # RGB
    RGB_Red = 0
    RGB_Green = 1
    RGB_Blue = 2
    # HSV
    HSV_Hue = 3
    HSV_Saturation = 4
    HSV_Value = 5
    # LAB
    LAB_L = 6
    LAB_a = 7
    LAB_b = 8
    # YUV
    YUV_Luma = 9 # Y
    YUV_U = 10
    YUV_V = 11
    # YCrCb
    YCrCb_Luma = 12 # Y
    YCrCb_Cr = 13
    YCrCb_Cb = 14
    # HLS
    HLS_Hue = 15
    HLS_Lightness = 16
    HLS_Saturation = 17
    # LUV
    LUV_L = 18
    LUV_U = 19
    LUV_V = 20
    # XYZ
    XYZ_X = 21
    XYZ_Y = 22
    XYZ_Z = 23
    
    
    @property
    def colorspace(self):
        """
        Get the colorspace for the current instance
        """
        return Colorspace(self.value // 3)
    
    @property
    def channel(self):
        """
        Get the channel number for the colorspace (0 to 2)
        """
        return self.value % 3


# Arguments to convert BGR to another colorspace
_colorspace_cvt = {
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
    Colorspace.RGB: cv2.COLOR_RGB2BGR,
    Colorspace.HSV: cv2.COLOR_HSV2BGR,
    Colorspace.LAB: cv2.COLOR_LAB2BGR,
    Colorspace.YUV: cv2.COLOR_YUV2BGR,
    Colorspace.YCrCb: cv2.COLOR_YCrCb2BGR,
    Colorspace.HLS: cv2.COLOR_HLS2BGR,
    Colorspace.LUV: cv2.COLOR_LUV2BGR,
    Colorspace.XYZ: cv2.COLOR_XYZ2BGR
}