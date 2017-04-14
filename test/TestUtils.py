#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import io
from numpy.testing import assert_approx_equal, assert_allclose, assert_array_equal
from nose.tools import assert_equal, assert_true, assert_false, assert_greater, assert_less
import cv2
import cv_algorithms
import numpy as np

class TestSpreadToGrayscale(object):
    def test_zero_input(self):
        i = np.zeros((10,10), dtype=np.float)
        o = np.zeros((10,10), dtype=np.uint8)
        assert_array_equal(o, cv_algorithms.spread_to_grayscale(i))

    def test_nonzero_input(self):
        i = np.zeros((10,10), dtype=np.float)
        i[3,5] = 17.25 
        o = np.zeros((10,10), dtype=np.uint8)
        o[3,5] = 255
        assert_array_equal(o, cv_algorithms.spread_to_grayscale(i))
