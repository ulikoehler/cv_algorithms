#include "common.hpp"
#include <stdlib.h>
#include <assert.h>
#include <stdbool.h>
#include <limits.h>
#include <string.h>
#include <stdint.h>


//Forward declaration required due to CFFI's requirement to have unmangled symbols
extern "C" {
	CFFI_DLLEXPORT int guo_hall_thinning(uint8_t* binary_image, size_t width, size_t height);
	CFFI_DLLEXPORT int zhang_suen_thinning(uint8_t* binary_image, size_t width, size_t height);
}

 
/**
 * Perform a logical AND on a memory array (a), ANDing it with another array (b)
 * We expect this function to be optimized by the compiler
 * specifically for the platform in use.
 */
void bitwiseANDInPlace(uint8_t* a, const uint8_t* b, size_t size) {
	for (size_t i = 0; i < size; ++i) {
		a[i] &= b[i];
	}
}

/**
 * Performs a single iteration of the Guo-Hall algorithm.
 * See http://opencv-code.com/quick-tips/implementation-of-guo-hall-thinning-algorithm/
 * and the original paper http://dx.doi.org/10.1145/62065.62074 for details.
 *
 * Compared to the opencv-code.com implementation, we also count the number of
 * changes during the iteration in order to avoid the cv::absdiff() call and the
 * super-expensive whole-image (possibly multi-Mibibyte) copy to prev.
 */
int guo_hall_iteration(uint8_t* img, uint8_t* mask, size_t width, size_t height, bool oddIteration) {
	/** 
	 * Compared to
	 * http://opencv-code.com/quick-tips/implementation-of-guo-hall-thinning-algorithm/
	 * we compute the mask in an inverted way so we don't have to invert while performing
	 * the AND.
	 */
	int changed = 0;
	for (unsigned int y = 1; y < height - 1; y++) {
		for (unsigned int x = 1; x < width - 1; x++) {
			if(IMG_XY(img, x, y) == 0) continue;
			// In the paper, figure 1 lists which Px corresponds to which coordinate
			bool p2 = IMG_XY(img, x, y - 1);
			bool p3 = IMG_XY(img, x + 1, y - 1);
			bool p4 = IMG_XY(img, x + 1, y);
			bool p5 = IMG_XY(img, x + 1, y + 1);
			bool p6 = IMG_XY(img, x, y + 1);
			bool p7 = IMG_XY(img, x - 1, y + 1);
			bool p8 = IMG_XY(img, x - 1, y);
			bool p9 = IMG_XY(img, x - 1, y - 1);

			unsigned int N1 = (p9 || p2) + (p3 || p4) + (p5 || p6) + (p7 || p8);
			unsigned int N2 = (p2 || p3) + (p4 || p5) + (p6 || p7) + (p8 || p9);
			unsigned int N = N1 < N2 ? N1 : N2;
			unsigned int m = 
				oddIteration ? (p8 && (p6 || p7 || !p9))
						     : (p4 && (p2 || p3 || !p5));
			unsigned int C =
				((!p2 && (p3 || p4)) +
				 (!p4 && (p5 || p6)) +
				 (!p6 && (p7 || p8)) +
				 (!p8 && (p9 || p2)));
			if (C == 1 && N >= 2 && N <= 3 && m == 0)   {
				//See above - mask is computed in an inverted waay
				IMG_XY(mask, x, y) = 0;
				changed++;
			}
		}
	}
	bitwiseANDInPlace(img, mask, width * height);
	return changed;
}


/**
 * Performs a single iteration of the Zhang-Suen algorithm.
 * See http://opencv-code.com/quick-tips/implementation-of-thinning-algorithm-in-opencv/
 * and the original paper https://dx.doi.org/10.1145/357994.358023 for details.
 * 
 * This function is very similar to the Guo-Hall algorithm. See guo_hall_iteration() for more implementation details
 */
int zhang_suen_iteration(uint8_t* img, uint8_t* mask, size_t width, size_t height, bool oddIteration) {

	int changed = 0;
	for (unsigned int y = 1; y < height - 1; y++) {
		for (unsigned int x = 1; x < width - 1; x++) {
			if(IMG_XY(img, x, y) == 0) continue;
			// In the Guo-Hall paper, figure 1 lists which Px corresponds to which coordinate
			bool p2 = IMG_XY(img, x, y - 1);
			bool p3 = IMG_XY(img, x + 1, y - 1);
			bool p4 = IMG_XY(img, x + 1, y);
			bool p5 = IMG_XY(img, x + 1, y + 1);
			bool p6 = IMG_XY(img, x, y + 1);
			bool p7 = IMG_XY(img, x - 1, y + 1);
			bool p8 = IMG_XY(img, x - 1, y);
			bool p9 = IMG_XY(img, x - 1, y - 1);

			int A  = (p2 == 0 && p3 == 1) + (p3 == 0 && p4 == 1) + 
                     (p4 == 0 && p5 == 1) + (p5 == 0 && p6 == 1) + 
                     (p6 == 0 && p7 == 1) + (p7 == 0 && p8 == 1) +
                     (p8 == 0 && p9 == 1) + (p9 == 0 && p2 == 1);
            int B  = p2 + p3 + p4 + p5 + p6 + p7 + p8 + p9;
            int m1 = oddIteration ? (p2 * p4 * p8) : (p2 * p4 * p6);
            int m2 = oddIteration ? (p2 * p6 * p8) : (p4 * p6 * p8);

			if (A == 1 && (B >= 2 && B <= 6) && m1 == 0 && m2 == 0)   {
				IMG_XY(mask, x, y) = 0; //Inverted mask!
				changed++;
			}
		}
	}
	bitwiseANDInPlace(img, mask, width * height);
	return changed;
}

/**
 * Main Guo-Hall thinning function (optimized).
 * See guo_hall_iteration() for more documentation.
 */
CFFI_DLLEXPORT int guo_hall_thinning(uint8_t* binary_image, size_t width, size_t height) {
	/* return -1 if we can't allocate the memory for the mask, else 0 */
	uint8_t* mask = (uint8_t*) malloc(width * height);
	if (mask == NULL) {
		return -1;
	}

	/**
	 * It is important to understand that with Guo-Hall black pixels will never get white.
	 * Therefore we don't need to reset the mask in each iteration.
	 * Especially for large images, this saves us many Mibibytes of memory transfer.
	 */
	memset(mask, UCHAR_MAX, width*height);

	int changed;
	do {
		changed =
			guo_hall_iteration(binary_image, mask, width, height, false) +
		    guo_hall_iteration(binary_image, mask, width, height, true);
	} while (changed != 0);

	//Cleanup
	free(mask);
	return 0;
}


/**
 * Main Zhang-Suen thinning function (optimized).
 * See guo_hall_thinning() for more documentation.
 */
CFFI_DLLEXPORT int zhang_suen_thinning(uint8_t* binary_image, size_t width, size_t height) {
	/* return -1 if we can't allocate the memory for the mask, else 0 */
	uint8_t* mask = (uint8_t*) malloc(width * height);
	if (mask == NULL) {
		return -1;
	}

	/**
	 * It is important to understand that with Guo-Hall black pixels will never get white.
	 * Therefore we don't need to reset the mask in each iteration.
	 * Especially for large images, this saves us many Mibibytes of memory transfer.
	 */
	memset(mask, UCHAR_MAX, width*height);

	int changed;
	do {
		changed =
			zhang_suen_iteration(binary_image, mask, width, height, false) +
		    zhang_suen_iteration(binary_image, mask, width, height, true);
	} while (changed != 0);

	//Cleanup
	free(mask);
	return 0;
}
