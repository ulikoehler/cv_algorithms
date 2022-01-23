#include "common.hpp"
#include <stdlib.h>
#include <assert.h>
#include <stdbool.h>
#include <limits.h>
#include <string.h>
#include <stdint.h>
#include <cmath>

//Forward declaration required due to CFFI's requirement to have unmangled symbols
extern "C" {
	CFFI_DLLEXPORT int pairwise_diff(const double* a, const double* b, double* result, size_t awidth, size_t bwidth);
}

CFFI_DLLEXPORT int pairwise_diff(const double* a, const double* b, double* result, size_t awidth, size_t bwidth) {

	//Iterate over all (a,b) element pairs
	for (size_t ax = 0; ax < awidth; ++ax) {
		for (size_t bx = 0; bx < bwidth; ++bx) {
			result[ax*awidth + bx] = std::abs((int64_t)(a[ax] - b[bx]));
		}	
	}
	return 0;
}
