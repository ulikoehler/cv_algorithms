#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Algorithms concerning resizing of images
"""
import numpy as np
import cv2

__all__ = ["resize_maintain_aspect_ratio"]

def resize_maintain_aspect_ratio(image, width=None, height=None):
    """
    Resize the given image while maintaining the aspect ratio.

    Parameters:
    - image: numpy.ndarray
        The input image to be resized.
    - width: int, optional
        The desired width of the resized image. If not specified, the height will be used to calculate the width.
    - height: int, optional
        The desired height of the resized image. If not specified, the width will be used to calculate the height.

    Returns:
    - numpy.ndarray
        The resized image.
    """
    # Get the original image dimensions
    original_height, original_width = image.shape[:2]

    # Calculate the aspect ratio
    aspect_ratio = original_width / original_height

    if width is not None:
        # Calculate the new height based on the desired width and aspect ratio
        new_height = int(width / aspect_ratio)
        # Resize the image using the new dimensions
        resized_image = cv2.resize(image, (width, new_height))
    elif height is not None:
        # Calculate the new width based on the desired height and aspect ratio
        new_width = int(height * aspect_ratio)
        # Resize the image using the new dimensions
        resized_image = cv2.resize(image, (new_width, height))
    else:
        # If neither width nor height is specified, return the original image
        resized_image = image

    return resized_image