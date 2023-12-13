#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import io
from numpy.testing import assert_array_equal
import cv2
from cv_algorithms.contours import *
import numpy as np
import unittest

class TestContours(unittest.TestCase):
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

    def test_contour_mask(self):
        cnts = [np.asarray([[0,0],[1,0],[1,1],[0,1],[0,0]]), # area 1
                np.asarray([[0,0],[2,0],[2,2],[0,2],[0,0]]), # area 4
                ]
        # Test return shape
        img = np.zeros((10,10,3), dtype=np.uint8)
        self.assertEqual((10, 10), contour_mask(img, []).shape)
        self.assertEqual((10, 10), contour_mask(img, cnts[0]).shape)
        self.assertEqual((10, 10), contour_mask(img, cnts).shape)
        self.assertEqual((10, 10), contour_mask((10,10), []).shape)
        self.assertEqual((10, 10), contour_mask((10,10,3), []).shape)
        # Test return values
        res = contour_mask(img, cnts)
        self.assertFalse((res == 255).all())
        self.assertFalse((res == 0).all())
        # Test no contours
        res = contour_mask(img, [])
        self.assertFalse((res == 255).all())
        self.assertTrue((res == 0).all())
