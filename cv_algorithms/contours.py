#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Contour utilities
"""
import cv2
import numpy as np

__all__ = ["meanCenter", "scaleByRefpoint", "extractPolygonMask", "expandRectangle",
           "cropBorderFraction", "filter_min_area", "filter_max_area",
           "sort_by_area", "contour_mask"]


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


def extractPolygonMask(img, rotrect, invmask=False, is_convex=True):
    """
    Extract a potentially rotated polygon from an image without rotating anything.
    (Rotation will cause inaccuracies and interpolation and therefore is often
    not desirable)

    This works by first extracting the polygon bounding box from the image,
    then creating an equivalently sized mask. Then, the original (yet now origin-referenced)
    rotated rectangle is drawn into said mask in black.
    After than, the mask is ORed onto the img section (for invmask=False)
    or ANDed onto the img section (for invmask=True), resulting in masked areas
    to be white (invmask=False) or black (invmask=True).

    This function will work only with grayscale images.

    Parameters
    ----------
    img : numpy (x,y) array
        The image to extract from
    rotrect : A 4-contour e.g. as returned by cv2.boundingBox(cv2.minAreaRect())
        The rectangle to extract
    invmask : bool
        Set to True for black fill color.
        Set to false for white fill color.
    is_convex : bool
        Set to false if the given polgon may not be convex.
    """
    x, y, w, h = cv2.boundingRect(rotrect)
    # Extract image section
    imgsec = img[y:y+h, x:x+w]
    # Create new, equivalently sized image
    mask = np.full_like(imgsec, 0 if invmask else 255)
    # Reference rr to origin
    rect_mask = rotrect - np.asarray([x, y])
    # Draw rr in the mask in black so we can OR the mask later
    if is_convex:
        cv2.fillConvexPoly(mask, rect_mask, 0)
    else:
        cv2.fillPoly(mask, [rect_mask], 0)
    # Apply mask to image section
    if invmask:
        imgsec &= mask
    else:
        imgsec |= mask
    return imgsec


def expandRectangle(rect, xfactor=3, yfactor=3):
    """
    Takes a (x,y,w,h) rectangle tuple and returns a new bounding
    rectangle that is centered on the center of the origin rectangle,
    but has a width/height that is larger by a given factor.

    The returned coordinates are rounded to integers
    """
    x, y, w, h = rect
    # Horizontal expansion
    x -= ((xfactor - 1) / 2) * w
    w *= xfactor
    # Horizontal expansion
    y -= ((yfactor - 1) / 2) * h
    h *= yfactor
    return (int(round(x)), int(round(y)),
            int(round(w)), int(round(h)))


def cropBorderFraction(img, crop_left=.1, crop_right=.1, crop_top=.1, crop_bot=.1):
    """
    Crop a fraction of the image at its borders.
    For example, cropping 10% (.1) of a 100x100 image left border
    would result in the leftmost 10px to be cropped.

    The number of pixels to be cropped are computed based on the original
    image size.
    """
    w, h = img.shape[0], img.shape[1]
    nleft = int(round(crop_left * w))
    nright = int(round(crop_right * w))
    ntop = int(round(crop_top * h))
    nbot = int(round(crop_bot * h))
    return img[ntop:-nbot, nleft:-nright]

def filter_min_area(contours, min_area):
    """
    Filter a list of contours, requiring
    a polygon to have an area of >= min_area
    to pass the filter.

    Uses OpenCV's contourArea for fast area computation

    Returns a list of contours.
    """
    return list(filter(lambda cnt: cv2.contourArea(cnt) >= min_area,
        contours))

def filter_max_area(contours, max_area):
    """
    Filter a list of contours, requiring
    a polygon to have an area of <= max_area
    to pass the filter.

    Uses OpenCV's contourArea for fast area computation

    Returns a list of contours.
    """
    return list(filter(lambda cnt: cv2.contourArea(cnt) <= max_area,
        contours))

def sort_by_area(contours, reverse=False):
    """
    Sort a list of contours by area
    """
    return sorted(contours, key=cv2.contourArea, reverse=True)

def contour_mask(shape, cnt):
    """
    Generate a black-white mask of one or multiple contours

    Parameters
    ==========
    cnt : Numpy array of points (contour) or list of contours
        The contour(s) to mark. May overlap.
    img: Numpy array or (width,height) tuple
        The image or its shape (only the .shape property will be used)
        Contour parts are ignored if they are outside the image

    Returns
    =======
    A black-white np.uint8 image with the contour marked in 100% white
    """
    if isinstance(cnt, np.ndarray):
        cnt = [cnt]
    if isinstance(shape, np.ndarray):
        shape = shape.shape
    if len(shape) > 2:
        shape = (shape[0], shape[1])
    # Create mask image
    mask = np.zeros(shape, dtype=np.uint8)
    cv2.drawContours(mask, cnt, -1, (255,255,255), -1)
    return mask
