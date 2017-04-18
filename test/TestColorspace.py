#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import io
from numpy.testing import assert_approx_equal, assert_allclose, assert_array_equal
from nose.tools import assert_equal, assert_true, assert_false, assert_greater, assert_less
import cv2
import cv_algorithms
from cv_algorithms.colorspace import Colorspace, ColorspaceChannel
import numpy as np

class TestColorspace(object):
    def test_colorspace_channels(self):
        assert_equal((ColorspaceChannel.RGB_Red,
                      ColorspaceChannel.RGB_Green,
                      ColorspaceChannel.RGB_Blue),
            Colorspace.RGB.channels)
        assert_equal((ColorspaceChannel.XYZ_X,
                      ColorspaceChannel.XYZ_Y,
                      ColorspaceChannel.XYZ_Z),
            Colorspace.XYZ.channels)

    def test_channel_colorspace(self):
        assert_equal(Colorspace.RGB, ColorspaceChannel.RGB_Red.colorspace)
        assert_equal(Colorspace.RGB, ColorspaceChannel.RGB_Green.colorspace)
        assert_equal(Colorspace.RGB, ColorspaceChannel.RGB_Blue.colorspace)
        assert_equal(Colorspace.LAB, ColorspaceChannel.LAB_L.colorspace)
        assert_equal(Colorspace.LAB, ColorspaceChannel.LAB_b.colorspace)
        assert_equal(Colorspace.XYZ, ColorspaceChannel.XYZ_X.colorspace)
        assert_equal(Colorspace.XYZ, ColorspaceChannel.XYZ_Z.colorspace)

    def test_channel_name(self):
        assert_equal("Red", ColorspaceChannel.RGB_Red.channel_name)
        assert_equal("b", ColorspaceChannel.LAB_b.channel_name)

class TestColorspaceConversion(object):
    def test_convert_to_colorspace(self):
        img = np.zeros((10,10,3), np.uint8)
        # Just test if it raises
        cv_algorithms.convert_to_colorspace(img, Colorspace.BGR)
        cv_algorithms.convert_to_colorspace(img, Colorspace.LAB)
        cv_algorithms.convert_to_colorspace(img, Colorspace.LAB, Colorspace.XYZ)
    
    def test_extract_channel(self):
        img = np.zeros((10,10,3), np.uint8)
        # Just test if it raises
        cv_algorithms.extract_channel(img, ColorspaceChannel.LAB_L)
