#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import cv_algorithms
from cv_algorithms import Neighbours, Direction
import numpy as np
import unittest

class TestNeighbours(unittest.TestCase):
    def test_binary_neighbours_simple(self):
        """Test binary direction detection"""
        img = np.zeros((10,8), dtype=np.uint8)
        y, x = 5, 4
        img[5,4] = 255
        # Currently just test whether it crashes
        directions = cv_algorithms.binary_neighbours(img)
        print(directions)
        self.assertEqual(0, directions[0,0])
        # NOTE: Directions are inversed, this is not a mistake
        # To the pixel on the NW of the white pixel
        #  the white pixel is SE!
        # Center
        self.assertEqual(0, directions[5, 4])
        # NW of the white pixel
        self.assertEqual((y-1, x-1), Neighbours.northwest_coords(y, x))
        self.assertEqual((y-1, x-1), Neighbours.coords(Direction.NorthWest, y, x))
        self.assertEqual((1 << 7), directions[y-1, x-1])
        self.assertFalse(Neighbours.is_northwest(directions[y-1, x-1]))
        self.assertFalse(Neighbours.is_north(directions[y-1, x-1]))
        self.assertFalse(Neighbours.is_northeast(directions[y-1, x-1]))
        self.assertFalse(Neighbours.is_west(directions[y-1, x-1]))
        self.assertFalse(Neighbours.is_east(directions[y-1, x-1]))
        self.assertFalse(Neighbours.is_southwest(directions[y-1, x-1]))
        self.assertFalse(Neighbours.is_south(directions[y-1, x-1]))
        self.assertTrue(Neighbours.is_southeast(directions[y-1, x-1]))
        # N of the white pixel
        self.assertEqual((y-1, x), Neighbours.north_coords(y, x))
        self.assertEqual((y-1, x), Neighbours.coords(Direction.North, y, x))
        self.assertEqual((1 << 6), directions[y-1, x])
        self.assertFalse(Neighbours.is_northwest(directions[y-1, x]))
        self.assertFalse(Neighbours.is_north(directions[y-1, x]))
        self.assertFalse(Neighbours.is_northeast(directions[y-1, x]))
        self.assertFalse(Neighbours.is_west(directions[y-1, x]))
        self.assertFalse(Neighbours.is_east(directions[y-1, x]))
        self.assertFalse(Neighbours.is_southwest(directions[y-1, x]))
        self.assertTrue(Neighbours.is_south(directions[y-1, x]))
        self.assertFalse(Neighbours.is_southeast(directions[y-1, x]))
        # NE of the white pixel
        self.assertEqual((y-1, x+1), Neighbours.northeast_coords(y, x))
        self.assertEqual((y-1, x+1), Neighbours.coords(Direction.NorthEast, y, x))
        self.assertEqual((1 << 5), directions[y-1, x+1])
        self.assertFalse(Neighbours.is_northwest(directions[y-1, x+1]))
        self.assertFalse(Neighbours.is_north(directions[y-1, x+1]))
        self.assertFalse(Neighbours.is_northeast(directions[y-1, x+1]))
        self.assertFalse(Neighbours.is_west(directions[y-1, x+1]))
        self.assertFalse(Neighbours.is_east(directions[y-1, x+1]))
        self.assertTrue(Neighbours.is_southwest(directions[y-1, x+1]))
        self.assertFalse(Neighbours.is_south(directions[y-1, x+1]))
        self.assertFalse(Neighbours.is_southeast(directions[y-1, x+1]))
        # W of the white pixel
        self.assertEqual((y, x-1), Neighbours.west_coords(y, x))
        self.assertEqual((y, x-1), Neighbours.coords(Direction.West, y, x))
        self.assertEqual((1 << 4), directions[y, x-1])
        self.assertFalse(Neighbours.is_northwest(directions[y, x-1]))
        self.assertFalse(Neighbours.is_north(directions[y, x-1]))
        self.assertFalse(Neighbours.is_northeast(directions[y, x-1]))
        self.assertFalse(Neighbours.is_west(directions[y, x-1]))
        self.assertTrue(Neighbours.is_east(directions[y, x-1]))
        self.assertFalse(Neighbours.is_southwest(directions[y, x-1]))
        self.assertFalse(Neighbours.is_south(directions[y, x-1]))
        self.assertFalse(Neighbours.is_southeast(directions[y, x-1]))
        # E of the white pixel
        self.assertEqual((y, x+1), Neighbours.east_coords(y, x))
        self.assertEqual((y, x+1), Neighbours.coords(Direction.East, y, x))
        self.assertEqual((1 << 3), directions[y, x+1])
        self.assertFalse(Neighbours.is_northwest(directions[y, x+1]))
        self.assertFalse(Neighbours.is_north(directions[y, x+1]))
        self.assertFalse(Neighbours.is_northeast(directions[y, x+1]))
        self.assertTrue(Neighbours.is_west(directions[y, x+1]))
        self.assertFalse(Neighbours.is_east(directions[y, x+1]))
        self.assertFalse(Neighbours.is_southwest(directions[y, x+1]))
        self.assertFalse(Neighbours.is_south(directions[y, x+1]))
        self.assertFalse(Neighbours.is_southeast(directions[y, x+1]))
        # SW of the white pixel
        self.assertEqual((y+1, x-1), Neighbours.southwest_coords(y, x))
        self.assertEqual((y+1, x-1), Neighbours.coords(Direction.SouthWest, y, x))
        self.assertEqual((1 << 2), directions[y+1, x-1])
        self.assertFalse(Neighbours.is_northwest(directions[y+1, x-1]))
        self.assertFalse(Neighbours.is_north(directions[y+1, x-1]))
        self.assertTrue(Neighbours.is_northeast(directions[y+1, x-1]))
        self.assertFalse(Neighbours.is_west(directions[y+1, x-1]))
        self.assertFalse(Neighbours.is_east(directions[y+1, x-1]))
        self.assertFalse(Neighbours.is_southwest(directions[y+1, x-1]))
        self.assertFalse(Neighbours.is_south(directions[y+1, x-1]))
        self.assertFalse(Neighbours.is_southeast(directions[y+1, x-1]))
        # S of the white pixel
        self.assertEqual((y+1, x), Neighbours.south_coords(y, x))
        self.assertEqual((y+1, x), Neighbours.coords(Direction.South, y, x))
        self.assertEqual((1 << 1), directions[y+1, x])
        self.assertFalse(Neighbours.is_northwest(directions[y+1, x]))
        self.assertTrue(Neighbours.is_north(directions[y+1, x]))
        self.assertFalse(Neighbours.is_northeast(directions[y+1, x]))
        self.assertFalse(Neighbours.is_west(directions[y+1, x]))
        self.assertFalse(Neighbours.is_east(directions[y+1, x]))
        self.assertFalse(Neighbours.is_southwest(directions[y+1, x]))
        self.assertFalse(Neighbours.is_south(directions[y+1, x]))
        self.assertFalse(Neighbours.is_southeast(directions[y+1, x]))
        # SE of the white pixel
        self.assertEqual((y+1, x+1), Neighbours.southeast_coords(y, x))
        self.assertEqual((y+1, x+1), Neighbours.coords(Direction.SouthEast, y, x))
        self.assertEqual((1 << 0), directions[y+1, x+1])
        self.assertTrue(Neighbours.is_northwest(directions[y+1, x+1]))
        self.assertFalse(Neighbours.is_north(directions[y+1, x+1]))
        self.assertFalse(Neighbours.is_northeast(directions[y+1, x+1]))
        self.assertFalse(Neighbours.is_west(directions[y+1, x+1]))
        self.assertFalse(Neighbours.is_east(directions[y+1, x+1]))
        self.assertFalse(Neighbours.is_southwest(directions[y+1, x+1]))
        self.assertFalse(Neighbours.is_south(directions[y+1, x+1]))
        self.assertFalse(Neighbours.is_southeast(directions[y+1, x+1]))

    def test_binary_neighbours_corner(self):
        # Just test if it crashes for something in the corners
        img = np.zeros((10,8), dtype=np.uint8)
        img[9,7] = 255
        img[0,0] = 255
        cv_algorithms.binary_neighbours(img)

    def test_direction_str(self):
        self.assertEqual("↑", str(Direction.North))
        self.assertEqual(Direction.North, Direction.from_unicode("↑"))
        self.assertEqual([Direction.SouthEast, Direction.North], Direction.from_unicode("↘↑"))