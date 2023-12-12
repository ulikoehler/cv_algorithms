#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import numpy as np
from numpy.testing import assert_array_equal
import cv_algorithms

class TestSpreadToGrayscale(object):
    def test_zero_input(self):
        i = np.zeros((10,10), dtype=float)
        o = np.zeros((10,10), dtype=np.uint8)
        assert_array_equal(o, cv_algorithms.spread_to_grayscale(i))

    def test_nonzero_input(self):
        i = np.zeros((10,10), dtype=float)
        i[3,5] = 17.25 
        o = np.zeros((10,10), dtype=np.uint8)
        o[3,5] = 255
        assert_array_equal(o, cv_algorithms.spread_to_grayscale(i))
