#include <stdlib.h>
#include <assert.h>
#include <stdbool.h>
#include <limits.h>
#include <string.h>
#include <stdint.h>

//Forward declaration required due to CFFI's requirement to have unmangled symbols
extern "C" {
	int xy_distance(const double* a, const double* b, double* result, size_t awidth, size_t bwidth);
	int nd_distance(const double* a, const double* b, double* result, size_t awidth, size_t bwidth, size_t height);
}

int xy_distance(const double* a, const double* b, double* result, size_t awidth, size_t bwidth) {

	//Iterate over all (a,b) element pairs
	for (size_t ax = 0; ax < awidth; ++ax) {
		for (size_t bx = 0; bx < bwidth; ++bx) {
			result[ax*awidth + bx] = abs(a[ax] - b[bx]);
		}	
	}
	return 0;
}

/**
 * Takes a list of m-d-coordinates (i.e. a (n,m) double array)
 */
int nd_distance(const double* a, const double* b, double* result, size_t awidth, size_t bwidth, size_t height) {

	//Iterate over all (a,b) element pairs
	for (size_t ax = 0; ax < awidth; ++ax) {
		for (size_t bx = 0; bx < bwidth; ++bx) {
			//Iterate over y
			for (size_t y = 0; y < height; ++y) {
				double aval = a[ax*awidth + y];
				double bval = b[bx*bwidth + y];
				result[ax*awidth*bwidth + bx*awidth + y] = abs(aval - bval);
			}
		}
	}
	return 0;
}