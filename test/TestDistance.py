#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import io
from numpy.testing import assert_approx_equal, assert_allclose, assert_array_equal
from nose.tools import assert_equal, assert_true, assert_false, assert_greater, assert_less
import cv2
import cv_algorithms
import numpy as np

class TestXYDistance(object):

    def testSimple(self):
        "Simple array-with-itself test"
        # Currently just run and see if it crashes
        a = np.asarray([1., 2., 3.])
        result = cv_algorithms.xy_distance(a, a)
        assert_allclose(result, [[0, 1, 2], [1, 0, 1], [2, 1, 0]])

    def testDifferentArrays(self):
        # Currently just run and see if it crashes
        a = np.asarray([1., 2., 3.])
        b = np.asarray([2., 3., 4.])
        result = cv_algorithms.xy_distance(a, b)
        assert_allclose(result, [[1, 2, 3], [0, 1, 2], [1, 0, 1]])
