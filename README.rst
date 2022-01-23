========
cv_algorithms
========
.. image:: https://circleci.com/gh/ulikoehler/cv_algorithms/tree/master.svg?style=svg
    :target: https://circleci.com/gh/ulikoehler/cv_algorithms/tree/master

A Python package (Python3 ready!) that contains implementations of various OpenCV algorithms are are not
available in OpenCV or OpenCV-contrib. This package is intended to be used with OpenCV 3.

Some performance-critical algorithms are written in optimized C code. The C code is accessed using `cffi <https://cffi.readthedocs.io/en/latest/>`_.

Currently implemented: 
 
 - Morphological algorithms
    - Guo-Hall thinning (C-optimized)
    - Zhang-Suen thinning (C-optimized)
    - Difference-of-Gaussian transform
 - Algorithms on contours
    - Masking extraction of convex polygon area from image without rotation
    - Scale around reference point or center
    - Fast computation of center by coordinate averaging
    - Center-invariant rescaling of upright bounding rectangle by x/y factors 
    - Filter by min/max area
    - Sort by area
    - Create binary contour mask
    - Grassfire transform
 - Colorspace metrics & utilities:
    - Convert image to any colorspace supported by OpenCV
    - Extract any channel from any colorspace directly
    - Euclidean RGB distance
 - Other structural algorithms
    - Which neighboring pixels are set in a binary image?
 - Algorithms on text rendering
    - Center text at coordinates
    - Auto-scale text to fix into box
 - Other algorithms
    - Remove n percent of image borders
    - Popcount (number of one bits) for 8, 16, 32 and 64 bit numpy arrays

As `cv2` represents images as `numpy <http://www.numpy.org/>`_ arrays, most algorithms generically work with numpy arrays.

Installation
============

Install the *stable* version

.. code-block:: bash

    $ sudo pip install cv_algorithms

On *Windows*:

.. code-block:: bash

    pip install cv_algorithms

Additionally, you need to :install OpenCV:`https://techoverflow.net/2022/01/23/how-to-fix-python-modulenotfounderror-no-module-named-cv2-on-windows/` if not already present.

Install the *bleeding-edge* version from GitHub

.. code-block:: bash

    # Python2
    $ sudo pip install git+https://github.com/ulikoehler/cv_algorithms.git
    # or (Python3)
    $ sudo pip3 install git+https://github.com/ulikoehler/cv_algorithms.git

Usage
=====

`Difference of Gaussian transform documentation & example <https://github.com/ulikoehler/cv_algorithms/blob/master/doc/DoG.md>`_

`Grassfire transform documentation & example <https://github.com/ulikoehler/cv_algorithms/blob/master/doc/Grassfire.md>`_

`Thinning documentation & example <https://github.com/ulikoehler/cv_algorithms/blob/master/doc/Thinning.md>`_

Here's a simple usage showcase:

.. code-block:: python

    import cv_algorithms
    # img must be a binary, single-channel (grayscale) image.
    thinned = cv_algorithms.guo_hall(img)

Contributions
=============

Contributions of any shape or form are welcome. Please submit a pull request or file an issue on GitHub.

Copyright (c) 2016 Uli Köhler <ukoehler@techoverflow.net>
