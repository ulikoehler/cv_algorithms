#!/usr/bin/env python3
import numpy as np

__all__ = ["__check_image_c_order", "__check_image_grayscale_2d",
           "__check_image_min_wh", "__check_array_uint8",
           "force_c_order_contiguous"]


def __check_image_min_wh(img, min_width, min_height):
    """Raise if the image does not have a given minimum width and height"""
    height, width = img.shape
    if height < min_height or width < min_width:
        raise ValueError("Thinning algorithm needs an image at least 3px wide and 3px high but size is {}".format(img.shape))


def __check_image_c_order(img):
    """Raise if the image is not in FORTRAN memory order"""
    # Check array memory order
    if np.isfortran(img): # i.e. not C-ordered
        raise ValueError("cv_algorithms works only on C-ordered arrays")
    if not img.flags['C_CONTIGUOUS']:
        raise ValueError("cv_algorithms works only on contiguous arrays")


def force_c_order_contiguous(img):
    """Raise if the image is not in FORTRAN memory order"""
    # Check array memory order
    if not img.flags['C_CONTIGUOUS']: # i.e. not C-ordered
        return np.ascontiguousarray(img)
    return img


def __check_image_grayscale_2d(img):
    """Raise if the image  is not a 2D image"""
    nd = len(img.shape)
    if nd == 3:
        raise ValueError("Can only use binary (i.e. grayscale) images")
    if nd != 2:
        raise ValueError("Image has wrong number of dimensions ({0} instead of 2)".format(nd))  

def __check_array_uint8(img):
    """Raise if the image  is not a 2D image"""
    if img.dtype != np.uint8:
        raise ValueError("Can only use images that have np.uint8 dtype")
