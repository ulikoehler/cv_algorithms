#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import io
from numpy.testing import assert_approx_equal, assert_allclose, assert_array_equal
from nose.tools import assert_equal, assert_true, assert_false, assert_greater, assert_less
import cv2
import cv_algorithms
import numpy as np

class TestGrassfire(object):
    def test_grassfire(self):
        "Test grassfire transform"
        mask = np.zeros((10,10), dtype=np.uint8)
        # Currently just test whether it crashes
        cv_algorithms.grassfire(mask)