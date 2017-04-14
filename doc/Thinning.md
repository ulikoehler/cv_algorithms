# Thinning

Thinning algorithms reduce a pixel-discrete binary structure to its [skeleton](https://en.wikipedia.org/wiki/Topological_skeleton).

`cv_algorithms` provides an easy-to-use implementation for Python & OpenCV. Due to it being implemented in C, it is suitable for high-performance applications.

Both the Zhang-Suen and the Guo-Hall variants are implemented. They only differ in some details of the calculation.

For a simple example source code see [this file](https://github.com/ulikoehler/cv_algorithms/blob/master/examples/thinning.py).

## Result example

This example has been generated using the example script linked to above.

Input (generated with GIMP fractal generator):

![Thinning input](https://github.com/ulikoehler/cv_algorithms/blob/master/examples/thinning-example.png)

Zhang-Suen output:

![Zhang-Suen output](https://raw.githubusercontent.com/ulikoehler/cv_algorithms/master/examples/zhang-suen-result.png)

Guo-Hall output:

![Guo-Hall output](https://raw.githubusercontent.com/ulikoehler/cv_algorithms/master/examples/guo-hall-result.png)
