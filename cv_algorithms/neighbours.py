#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Thinning algorithms
"""
import numpy as np
from ._ffi import *
from ._checks import *
import enum

__all__ = ["binary_neighbours", "Neighbours", "Direction"]

_ffi.cdef('''
int binary_neighbours(uint8_t* dst, const uint8_t* src, int width, int height);
''')

def binary_neighbours(img):
    """
    Takes a binary image and, for each pixel, computes
    which surrounding pixels are non-zero.
    Depdending on those pixels, bits in the uint8 output
    array are set or unset

    Parameters
    ==========
    img : numpy array-like
        A grayscale image that is assumed to be binary
        (every non-zero value is interpreted as 0).
        Usually this is a pre-thinned image.

    Returns
    =======
    A uint8-type output array the same shape of img,
    where the following bits are set or unset,
    if the respective neighbouring pixel is set or unset.

    Bit index by position:

        0 1 2
        3   4
        5 6 7

    Note that for Numpy due to the coordinate system,
    the respective pixels can be accessed like this:

        [y-1,x-1]  [y-1,x]  [y-1,x+1]
        [y,x-1]    [y,x]    [y,x+1]
        [y+1,x-1]  [y+1,x]  [y+1,x+1]

    This is equivalent to the coordinate system when displaying
    the image using matplotlib imshow

    The positions in this matrix correspond to the bit number
    shown above, e.g. bit #4 is (1 << 4) ORed to the result.
    """
    # Check if image has the correct type
    __check_image_grayscale_2d(img)
    img = force_c_order_contiguous(img)
    __check_array_uint8(img)

    height, width = img.shape

    # Allocate output array
    # uint32 is used so there is no overflow for large inputs
    out = np.zeros(img.shape, dtype=np.uint8, order="C")
    assert not np.isfortran(out)

    # Extract pointer to binary data
    srcptr = _ffi.cast("uint8_t*", img.ctypes.data)
    dstptr = _ffi.cast("uint8_t*", out.ctypes.data)

    rc = _libcv_algorithms.binary_neighbours(dstptr, srcptr, width, height)
    if rc != 0:
        raise ValueError("Internal error (return code {0}) in algorithm C code".format(rc))
    return out

class Direction(enum.IntEnum):
    """
    Direction enum, mostly used as an argument for various functions
    """
    NorthWest = 1
    North = 2
    NorthEast = 3
    West = 4
    East = 5
    SouthWest = 6
    South = 7
    SouthEast = 8

    def __str__(self):
        return {
            Direction.West: "←",
            Direction.North: "↑",
            Direction.East: "→",
            Direction.South: "↓",
            Direction.NorthWest: "↖",
            Direction.NorthEast: "↗",
            Direction.SouthEast: "↘",
            Direction.SouthWest: "↙"
        }[self]

    @staticmethod
    def opposite(self):
        """
        Return the opposite direction
        """
        return {
            Direction.West: Direction.East,
            Direction.North: Direction.South,
            Direction.East: Direction.West,
            Direction.South: Direction.North,
            Direction.NorthWest: Direction.SouthEast,
            Direction.NorthEast: Direction.SouthWest,
            Direction.SouthEast: Direction.NorthWest,
            Direction.SouthWest: Direction.NorthEast
        }[self]

    @staticmethod
    def from_unicode(s):
        """
        Convert a arrow string (such as returned by __str__())
        to one or multiple Direction instances
        """
        if len(s) == 1:
            return {
                "←": Direction.West,
                "↑": Direction.North,
                "→": Direction.East,
                "↓": Direction.South,
                "↖": Direction.NorthWest,
                "↗": Direction.NorthEast,
                "↘": Direction.SouthEast,
                "↙": Direction.SouthWest
            }[s]
        else:
            return [Direction.from_unicode(c) for c in s]


class Neighbours():
    """
    *is_xxx():*
    Methods for checking one pixel of the result of binary_neighbours()
    if it has a marked neighbour for a given result.

    *xxx_coords():*
    Get the numpy coordinate for the pixel in the given
    direction of the given coordinate
    """
    @staticmethod
    def is_northwest(pixel): return bool(pixel & (1 << 0))
    @staticmethod
    def is_north(pixel): return bool(pixel & (1 << 1))
    @staticmethod
    def is_northeast(pixel): return bool(pixel & (1 << 2))
    @staticmethod
    def is_west(pixel): return bool(pixel & (1 << 3))
    @staticmethod
    def is_east(pixel): return bool(pixel & (1 << 4))
    @staticmethod
    def is_southwest(pixel): return bool(pixel & (1 << 5))
    @staticmethod
    def is_south(pixel): return bool(pixel & (1 << 6))
    @staticmethod
    def is_southeast(pixel): return bool(pixel & (1 << 7))

    @staticmethod
    def is_direction(direction, pixel):
        return {
            Direction.NorthEast: Neighbours.is_northeast,
            Direction.North: Neighbours.is_north,
            Direction.East: Neighbours.is_east,
            Direction.NorthWest: Neighbours.is_northwest,
            Direction.SouthEast: Neighbours.is_southeast,
            Direction.South: Neighbours.is_south,
            Direction.West: Neighbours.is_west,
            Direction.SouthWest: Neighbours.is_southwest
        }[direction](pixel)

    @staticmethod
    def northwest_coords(y, x): return (y-1, x-1)
    @staticmethod
    def north_coords(y, x): return (y-1, x)
    @staticmethod
    def northeast_coords(y, x): return (y-1, x+1)
    @staticmethod
    def west_coords(y, x): return (y, x-1)
    @staticmethod
    def east_coords(y, x): return (y, x+1)
    @staticmethod
    def southwest_coords(y, x): return (y+1, x-1)
    @staticmethod
    def south_coords(y, x): return (y+1, x)
    @staticmethod
    def southeast_coords(y, x): return (y+1, x+1)

    @staticmethod
    def coords(direction, y, x):
        return {
            Direction.NorthEast: Neighbours.northeast_coords,
            Direction.North: Neighbours.north_coords,
            Direction.East: Neighbours.east_coords,
            Direction.NorthWest: Neighbours.northwest_coords,
            Direction.SouthEast: Neighbours.southeast_coords,
            Direction.South: Neighbours.south_coords,
            Direction.West: Neighbours.west_coords,
            Direction.SouthWest: Neighbours.southwest_coords
        }[direction](y, x)

    @staticmethod
    def iterate_directions(dirs):
        """
        Iterate elements of the Direction IntEnum
        if the corresponding bit is set 

        See binary_directions for a definition of the
        bits.

        The order to the directions is the same as the bit order
        """
        return (direction for direction in Direction
            if Neighbours.is_direction(direction, dirs))
    
