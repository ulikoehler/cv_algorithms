#!/usr/bin/env python3
# -*- coding: utf-8 -*-
from numpy.testing import assert_allclose, assert_array_equal
import cv2
import cv_algorithms
import numpy as np
import unittest

class TestXYDistance(unittest.TestCase):
    def test_simple(self):
        "Simple array-with-itself test"
        # Currently just run and see if it crashes
        a = np.asarray([1., 2., 3.])
        result = cv_algorithms.pairwise_diff(a, a)
        assert_allclose(result, [[0, 1, 2], [1, 0, 1], [2, 1, 0]])

    def test_different_arrays(self):
        # Currently just run and see if it crashes
        a = np.asarray([1., 2., 3.])
        b = np.asarray([2., 3., 4.])
        result = cv_algorithms.pairwise_diff(a, b)
        assert_allclose(result, [[1, 2, 3], [0, 1, 2], [1, 0, 1]])

class TestColorspaceDistance(unittest.TestCase):
    def test_rgb_distance(self):
        img = np.zeros((10,10,3))
        # black has zero distance to itself
        exp = np.zeros((10,10))
        assert_array_equal(exp, cv_algorithms.rgb_distance(img, (0,0,0)))
        # Single channel
        exp = np.full((10,10), 5)
        assert_array_equal(exp, cv_algorithms.rgb_distance(img, (5,0,0)))
        # Multiple channel
        exp = np.full((10,10), 5)
        assert_array_equal(exp, cv_algorithms.rgb_distance(img, (3,0,4)))
        # Single pixel
        img[5,4] = (3,0,4)
        exp = np.full((10,10), 5)
        exp[5,4] = 0.
        assert_array_equal(exp, cv_algorithms.rgb_distance(img, (3,0,4)))