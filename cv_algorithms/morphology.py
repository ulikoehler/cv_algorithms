#!/usr/bin/env python3
import cv2

__all__ = ["difference_of_gaussian"]

def difference_of_gaussian(img, ksize1, ksize2, invert=False, normalize=True):
    img1 = cv2.GaussianBlur(img, (ksize1, ksize1), 0)
    img2 = cv2.GaussianBlur(img, (ksize2, ksize2), 0)
    dog = cv2.subtract(img1, img2)
    # Normalize
    if normalize:
        dog = cv2.normalize(dog, None, 0, 255, cv2.NORM_MINMAX, cv2.CV_8U)
    if invert:
        dog = 255 - dog
    return dog
