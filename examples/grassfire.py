#!/usr/bin/env python3
"""
cv_algorithms grassfire example
"""
import cv2
import cv_algorithms

# Read example file which contains an example binary image.
img = cv2.imread("grassfire-example.png")
# Convert to grayscale
# This is only required because OpenCV always reads images as color
imgGray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

# Alternate algorithm (but very similar)
# Only slight differences in the output!
result = cv_algorithms.grassfire(imgGray)

# The result contains 32-bit counters
# In order to write it to an output image,
#  we need to convert it to (0-255) grayscale.
result = cv_algorithms.spread_to_grayscale(result)

# Write to file so you can see what's been done
cv2.imwrite("grassfire-result.png", result)
