# cv_algorithms

![image](https://circleci.com/gh/ulikoehler/cv_algorithms/tree/master.svg?style=svg)
A Python package (Python3 ready!) that contains implementations of various OpenCV algorithms are are not
available in OpenCV or OpenCV-contrib. This package is intended to be used with OpenCV 3.

Some performance-critical algorithms are written in optimized C code. The C code is accessed using [cffi](https://cffi.readthedocs.io/en/latest/)
Currently implemented:
-   Morphological algorithms
    -   [Guo-Hall thinning (C-optimized)](https://github.com/ulikoehler/cv_algorithms/blob/master/doc/Thinning.md)
    -   [Zhang-Suen thinning (C-optimized)](https://github.com/ulikoehler/cv_algorithms/blob/master/doc/Thinning.md)
    -   [Difference-of-Gaussian transform](https://github.com/ulikoehler/cv_algorithms/blob/master/doc/DoG.md)
-   Algorithms on contours
    -   Masking extraction of convex polygon area from image without rotation
    -   Scale around reference point or center
    -   Fast computation of center by coordinate averaging
    -   Center-invariant rescaling of upright bounding rectangle by x/ factors
    -   Filter by min/max area
    -   Sort by area
    -   Create binary contour mask
    -   [Grassfire transform](https://github.com/ulikoehler/cv_algorithms/blob/master/doc/Grassfire.md)
-   Colorspace metrics & utilities:
    -   Convert image to any colorspace supported by OpenCV
    -   Extract any channel from any colorspace directly
    -   Euclidean RGB distance
-   Other structural algorithms
    -   Which neighboring pixels are set in a binary image?
-   Algorithms on text rendering
    -   Center text at coordinates
    -   Auto-scale text to fix into box
-   Other algorithms
    -   Remove n percent of image borders
    -   Popcount (number of one bits) for 8, 16, 32 and 64 bit numpy arrays
    -   Resize an image, maintaining the aspect ratio

As OpenCV's Python bindings (`cv2`) represents images as [numpy](http://www.numpy.org/) arrays, most algorithms generically work with *numpy*1  arrays.

## Installation

Install the *stable* version:

``` {.sourceCode .bash}
pip install cv_algorithms
```

How to install the *bleeding-edge* version from GitHub

``` {.sourceCode .bash}
pip install git+https://github.com/ulikoehler/cv_algorithms.git
```

How to *build yourself* - we use [Poetry](https://python-poetry.org/):
```sh
poetry build
```

Potentially, you need to [install OpenCV](https://techoverflow.net/2022/01/23/how-to-fix-python-modulenotfounderror-no-module-named-cv2-on-windows/) if not already present. I recommend first trying to install without that, since modern Python versions will take care of that automatically.

## Usage

[Difference of Gaussian transform documentation & example](https://github.com/ulikoehler/cv_algorithms/blob/master/doc/DoG.md)
[Grassfire transform documentation & example](https://github.com/ulikoehler/cv_algorithms/blob/master/doc/Grassfire.md)
[Thinning documentation & example](https://github.com/ulikoehler/cv_algorithms/blob/master/doc/Thinning.md)

Here\'s a simple usage showcase:

``` {.sourceCode .python}
import cv_algorithms
# img must be a binary, single-channel (grayscale) image.
thinned = cv_algorithms.guo_hall(img)
```

## Contributions

Contributions of any shape or form are welcome. Please submit a pull
request or file an issue on GitHub.

Copyright (c) 2016-2022 Uli Köhler \<<cv_algorithms@techoverflow.net>\>
