#!/usr/bin/env python3
"""
cv_algorithms thinning example
"""
import cv2
import cv_algorithms

# Read example file which contains some fractals (generated by GIMP fractal renderer)
img = cv2.imread("thinning-example.png")
# Convert to grayscale
imgGray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

# Difference of Gaussian
result = cv_algorithms.difference_of_gaussian(imgGray, 5, 1, invert=True)

# Write to file so you can see what's been done
cv2.imwrite("difference-of-gaussian-result.png", result)
