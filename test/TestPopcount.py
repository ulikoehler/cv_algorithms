#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import io
from numpy.testing import assert_approx_equal, assert_allclose, assert_array_equal
from nose.tools import assert_equal, assert_true, assert_false, assert_greater, assert_less
import cv2
import cv_algorithms
import numpy as np

class TestPopcount(object):
    def test_popcount8(self):
        i = np.zeros((10,10), dtype=np.uint8)
        o = np.zeros((10,10), dtype=np.uint8)
        # No bits
        assert_array_equal(o, cv_algorithms.popcount(i))
        # One bit
        i[1,1] = 8
        o[1,1] = 1
        assert_array_equal(o, cv_algorithms.popcount(i))
        # Two bits
        i[1,1] = 3
        o[1,1] = 2
        assert_array_equal(o, cv_algorithms.popcount(i))
        # Two bits
        i[1,1] = 7
        o[1,1] = 3
        assert_array_equal(o, cv_algorithms.popcount(i))
        # All bits
        i[1,1] = 0xFF
        o[1,1] = 8
        assert_array_equal(o, cv_algorithms.popcount(i))

    def test_popcount16(self):
        i = np.zeros((10,10), dtype=np.uint16)
        o = np.zeros((10,10), dtype=np.uint8)
        # No bits
        assert_array_equal(o, cv_algorithms.popcount(i))
        # One bit
        i[1,1] = 8
        o[1,1] = 1
        assert_array_equal(o, cv_algorithms.popcount(i))
        # Two bits
        i[1,1] = 3
        o[1,1] = 2
        assert_array_equal(o, cv_algorithms.popcount(i))
        # Two bits
        i[1,1] = 7
        o[1,1] = 3
        assert_array_equal(o, cv_algorithms.popcount(i))
        # All bits
        i[1,1] = 0xFFFF
        o[1,1] = 16
        assert_array_equal(o, cv_algorithms.popcount(i))

    def test_popcount32(self):
        i = np.zeros((10,10), dtype=np.uint32)
        o = np.zeros((10,10), dtype=np.uint8)
        # No bits
        assert_array_equal(o, cv_algorithms.popcount(i))
        # One bit
        i[1,1] = 8
        o[1,1] = 1
        assert_array_equal(o, cv_algorithms.popcount(i))
        # Two bits
        i[1,1] = 3
        o[1,1] = 2
        assert_array_equal(o, cv_algorithms.popcount(i))
        # Two bits
        i[1,1] = 7
        o[1,1] = 3
        assert_array_equal(o, cv_algorithms.popcount(i))
        # All bits
        i[1,1] = 0xFFFFFFFF
        o[1,1] = 32
        assert_array_equal(o, cv_algorithms.popcount(i))

    def test_popcount64(self):
        i = np.zeros((10,10), dtype=np.uint64)
        o = np.zeros((10,10), dtype=np.uint8)
        # No bits
        assert_array_equal(o, cv_algorithms.popcount(i))
        # One bit
        i[1,1] = 8
        o[1,1] = 1
        assert_array_equal(o, cv_algorithms.popcount(i))
        # Two bits
        i[1,1] = 3
        o[1,1] = 2
        assert_array_equal(o, cv_algorithms.popcount(i))
        # Two bits
        i[1,1] = 7
        o[1,1] = 3
        assert_array_equal(o, cv_algorithms.popcount(i))
        # All bits
        i[1,1] = 0xFFFFFFFFFFFFFFFF
        o[1,1] = 64
        assert_array_equal(o, cv_algorithms.popcount(i))

