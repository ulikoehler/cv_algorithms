#include <stdlib.h>
#include <assert.h>
#include <stdbool.h>
#include <limits.h>
#include <string.h>
#include <stdint.h>

/** 
 * Dirty macro to directly access an image at X/Y coordinates.
 * Assumes that the width variable is defined to the width of the image
 */
#define XY3D(arr, x, y, z) arr[(x)* + (width)*(y)+]

/**
 * Takes a list of m-d-coordinates (i.e. a (n,m) double array)
 */
int xy_distance(const double* a, const double* b, double* result, size_t awidth, size_t bwidth, size_t height) {

	//Iterate over all (a,b) element pairs
	for (size_t ax = 0; ax < awidth; ++ax) {
		for (size_t bx = 0; bx < bwidth; ++bx) {
			//Iterate over y
			for (size_t y = 0; y < height; ++y) {
				double aval = a[ax*width + y];
				double bval = b[bx*width + y];
				result[ax*awidth*bwidth + bx*awidth + y] = abs(aval - bval);
			}
	}
	return 0;
}