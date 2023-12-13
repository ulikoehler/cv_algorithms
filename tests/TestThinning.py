#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import io
import cv2
import cv_algorithms
import numpy as np
import unittest

class TestThinning(unittest.TestCase):
    def setUp(self) -> None:
        img = cv2.imread("examples/thinning-example.png")
        # Convert to grayscale
        self.img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        self.img_thresh = cv2.threshold(self.img, 180, 255, cv2.THRESH_BINARY)[1]
        
        return super().setUp()

    def _checkThinningImage(self, result):
        # Check corner conditions for thinning algorithms
        # a) No pixels that were not white before should be white now...
        black_orig = 255 - self.img_thresh
        self.assertFalse(np.any(np.logical_and(black_orig, result)))
        # b) There are some white pixels, at least for this example image
        self.assertTrue(np.any(result == 255))
        # c) There are more black pixels than before
        orig_numblack = np.sum(self.img_thresh == 0)
        result_numblack = np.sum(result == 0)
        self.assertGreater(result_numblack, orig_numblack)

    def testGuoHall(self):
        "Test Guo-Hall thinning"
        # Currently just run and see if it crashes
        result = cv_algorithms.guo_hall(self.img_thresh)
        self._checkThinningImage(result)

    def testZhangSuen(self):
        "Test Zhang-Suen thinning"
        # Currently just run and see if it crashes
        result = cv_algorithms.zhang_suen(self.img_thresh)
        self._checkThinningImage(result)
