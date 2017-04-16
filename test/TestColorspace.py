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
