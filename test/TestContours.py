#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import io
from numpy.testing import assert_approx_equal, assert_allclose, assert_array_equal
from nose.tools import assert_equal, assert_true, assert_false, assert_greater, assert_less
import cv2
from cv_algorithms.contours import *
import numpy as np

def assert_approx_equal_list(a, b):
    assert_approx_equal(np.asarray(a), np.asarray(b))

class TestContours(object):
    def testAreaFilter(self):
        "Test contour area filter"
        cnts = [np.asarray([[0,0],[1,0],[1,1],[0,1],[0,0]]), # area 1
                np.asarray([[0,0],[2,0],[2,2],[0,2],[0,0]]), # area 4
                ]
        # Min area
        assert_array_equal(cnts, filter_min_area(cnts, 0.5))
        assert_array_equal(cnts, filter_min_area(cnts, 1.0))
        assert_array_equal([cnts[1]], filter_min_area(cnts, 1.1))
        assert_array_equal([cnts[1]], filter_min_area(cnts, 2.0))
        assert_array_equal([cnts[1]], filter_min_area(cnts, 3.0))
        assert_array_equal([cnts[1]], filter_min_area(cnts, 4.0))
        assert_array_equal([], filter_min_area(cnts, 5.0))
        # Max area
        assert_array_equal([], filter_max_area(cnts, 0.5))
        assert_array_equal([cnts[0]], filter_max_area(cnts, 1.0))
        assert_array_equal([cnts[0]], filter_max_area(cnts, 1.1))
        assert_array_equal([cnts[0]], filter_max_area(cnts, 2.0))
        assert_array_equal([cnts[0]], filter_max_area(cnts, 3.0))
        assert_array_equal(cnts, filter_max_area(cnts, 4.0))
        assert_array_equal(cnts, filter_max_area(cnts, 5.0))