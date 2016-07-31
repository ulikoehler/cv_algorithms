#!/usr/bin/env python3
import cv2
import cv_algorithms

# Read example file which contains some fractals (generated by GIMP fractal renderer)
img = cv2.imread("thinning-example.png")
# Convert to grayscale
imgGray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

# Thinning needs binary images (i.e. black/white, no gray!)
# 180 is a hand-tuned threshold for the example image.
# The WHITE parts will be trimmed, so depending on the image, you have to use
#  either cv2.THRESH_BINARY or cv2.THRESH_BINARY_INV
imgThresh = cv2.threshold(imgGray, 180, 255, cv2.THRESH_BINARY)[1]

# Perform thinning out-of-place
guo_hall = cv_algorithms.guo_hall(imgThresh)

# ... or allow the library to modify the original image (= faster):
# cv_algorithms.guo_hall_inplace(imgThresh, auto_bgr2gray=True)

# Write to file so you can see what's been done
cv2.imwrite("guo-hall-result.png", guo_hall)