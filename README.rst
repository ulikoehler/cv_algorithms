========
cv_algorithms
========

A Python package (Python3 ready!) that contains implementations of various OpenCV algorithms are are not
available in OpenCV or OpenCV-contrib. This package is intended to be used with OpenCV 3.

Most implementations are written in optimized C code. The C code is accessed using [cffi](https://cffi.readthedocs.io/en/latest/).

Currently implemented: 
 
 - Guo-Hall thinning
 - Zhang-Suen thinning

As `cv2` represents images as [`numpy`](http://www.numpy.org/) arrays, most algorithms generically work with numpy arrays.

Installation
============

.. code-block:: bash

    # Python2
    $ sudo pip install git+https://github.com/ulikoehler/cv_algorithms.git
    # or (Python3)
    $ sudo pip3 install git+https://github.com/ulikoehler/cv_algorithms.git


Usage
=====

[Full thinning example](https://github.com/ulikoehler/cv_algorithms/blob/master/examples/thinning.py)

.. code-block: python

    import cv_algorithms
    # img must be a binary, single-channel (grayscale) image.
    thinned = cv_algorithms.guo_hall(img)

Contributions
=============

Contributions of any shape or form are welcome. Please submit a pull request or file an issue on GitHub.

Copyright (c) 2016 Uli Köhler <ukoehler@techoverflow.net>