#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import cv_algorithms
from cv_algorithms.colorspace import Colorspace, ColorspaceChannel
import numpy as np
import unittest

class TestColorspace(unittest.TestCase):
    def test_colorspace_channels(self):
        self.assertEqual((ColorspaceChannel.RGB_Red,
                      ColorspaceChannel.RGB_Green,
                      ColorspaceChannel.RGB_Blue),
            Colorspace.RGB.channels)
        self.assertEqual((ColorspaceChannel.XYZ_X,
                      ColorspaceChannel.XYZ_Y,
                      ColorspaceChannel.XYZ_Z),
            Colorspace.XYZ.channels)

    def test_channel_colorspace(self):
        self.assertEqual(Colorspace.RGB, ColorspaceChannel.RGB_Red.colorspace)
        self.assertEqual(Colorspace.RGB, ColorspaceChannel.RGB_Green.colorspace)
        self.assertEqual(Colorspace.RGB, ColorspaceChannel.RGB_Blue.colorspace)
        self.assertEqual(Colorspace.LAB, ColorspaceChannel.LAB_L.colorspace)
        self.assertEqual(Colorspace.LAB, ColorspaceChannel.LAB_b.colorspace)
        self.assertEqual(Colorspace.XYZ, ColorspaceChannel.XYZ_X.colorspace)
        self.assertEqual(Colorspace.XYZ, ColorspaceChannel.XYZ_Z.colorspace)

    def test_channel_name(self):
        self.assertEqual("Red", ColorspaceChannel.RGB_Red.channel_name)
        self.assertEqual("b", ColorspaceChannel.LAB_b.channel_name)

class TestColorspaceConversion(unittest.TestCase):
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
