# Difference of Gaussian transform

The [Difference of Gaussian transform](https://en.wikipedia.org/wiki/Difference_of_Gaussians) is a standard algorithm for filtering an image to enhance features in an image, operating similarly to edge detectors.

`cv_algorithms` provides an easy-to-use implementation for Python & OpenCV. Although the algorithm is implemented in pure Python, all the computationally expensive parts of the algorithm are implemented by OpenCV or NumPy.

For a simple example source code see [this file](https://github.com/ulikoehler/cv_algorithms/blob/master/examples/difference-of-gaussian.py).

## Result example

This example has been generated using the example script linked to above.

Input (generated with GIMP fractal generator):

![Thinning input](https://github.com/ulikoehler/cv_algorithms/blob/master/examples/thinning-example.png)

Output:

![DoG output](https://raw.githubusercontent.com/ulikoehler/cv_algorithms/master/examples/difference-of-gaussian-result.png)
