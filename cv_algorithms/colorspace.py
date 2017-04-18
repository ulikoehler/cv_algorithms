#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Utilities for extracting color channels out of arbitrary images
"""
import cv2
import numpy as np
import enum
from .text import putTextAutoscale

__all__ = ["ColorspaceChannel", "Colorspace", "convert_to_colorspace",
    "extract_channel", "colorspace_components_overview"]


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

    @property
    def channel_name(self):
        """
        The name of the channel,
        not including the colorspace name.

        Example: RGB_Red => Red
        """
        return self.name.partition("_")[2]


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

def extract_channel(img, channel, source=Colorspace.BGR, as_rgb=False):
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
    as_rgb : bool
        Set to True to obtain the graysca
    Returns
    =======
    The resulting channel as a NumPy image.
    The returned array is similar to a grayscale image.
    """
    target_space = channel.colorspace
    # Convert to the correct colorspace
    img = convert_to_colorspace(img, target_space, source)
    # Extract appropriate channel
    gray = img[:,:,channel.channel_idx]
    return cv2.cvtColor(gray, cv2.COLOR_GRAY2RGB) if as_rgb else gray


def colorspace_components_overview(img):
    """
    Render an image that shows all channels of the given image
    in all colorspaces in an ordered and labeled manner.
    """
    height, width, _ = img.shape
    ncolspaces = len(Colorspace)
    hspace = int(0.1 * width)
    vspace = int(0.1 * height)
    textheight = int(0.3 * height)

    h = ncolspaces * height + vspace * (ncolspaces - 1) + textheight * ncolspaces
    w = width * 3 + hspace * 2
    out = np.full((h, w), 255, dtype=np.uint8)

    for i, colorspace in enumerate(Colorspace):
        # Compute offsets
        vofs = textheight * (i + 1) + (vspace + height) * i
        hofs = lambda col: hspace * col + width * col
        textvofs = vofs - textheight / 2
        texthofs = lambda col: hofs(col) + width / 2
        # Get channels of current colorspace
        channels = colorspace.channels
        # Channel text
        chn0txt = "{} {}".format(colorspace.name, colorspace.channels[0].channel_name)
        chn1txt = "{} {}".format(colorspace.name, colorspace.channels[1].channel_name)
        chn2txt = "{} {}".format(colorspace.name, colorspace.channels[2].channel_name)
        # Extract all channels and convert to gray RGB mge
        chn0 = extract_channel(img, channels[0])
        chn1 = extract_channel(img, channels[1])
        chn2 = extract_channel(img, channels[2])
        # Copy image channels to output
        out[vofs:vofs + height, hofs(0):hofs(0) + width] = chn0
        out[vofs:vofs + height, hofs(1):hofs(1) + width] = chn1
        out[vofs:vofs + height, hofs(2):hofs(2) + width] = chn2
        # Render text
        putTextAutoscale(out, chn0txt, (texthofs(0), textvofs),
            cv2.FONT_HERSHEY_COMPLEX, width, textheight, color=0)
        putTextAutoscale(out, chn1txt, (texthofs(1), textvofs),
            cv2.FONT_HERSHEY_COMPLEX, width, textheight, color=0)
        putTextAutoscale(out, chn2txt, (texthofs(2), textvofs),
            cv2.FONT_HERSHEY_COMPLEX, width, textheight, color=0)

    return out
