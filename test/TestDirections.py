#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import io
from numpy.testing import assert_approx_equal, assert_allclose, assert_array_equal
from nose.tools import assert_equal, assert_true, assert_false, assert_greater, assert_less
import cv2
import cv_algorithms
import numpy as np

class TestDirections(object):
    def test_binary_directions(self):
        img = np.zeros((10,10), dtype=np.uint8)
        img[5,4] = 255
        # Currently just test whether it crashes
        directions = cv_algorithms.binary_directions(img)
        print(directions)
        assert_equal(0, directions[0,0])
        assert_equal(1 << 5, directions[5,4])