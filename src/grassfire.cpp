#include "common.hpp"
#include <stdlib.h>
#include <assert.h>
#include <stdbool.h>
#include <limits.h>
#include <string.h>
#include <stdint.h>
#include <algorithm>


using std::min;
using std::max;

//Forward declaration required due to CFFI's requirement to have unmangled symbols
extern "C" {
    CFFI_DLLEXPORT int grassfire(uint32_t* dst, const uint8_t* mask, int width, int height);
}

/**
 * Fast C implementation of the grassfire algorithm.
 * Takes a destination counter array (must be zero-initialized)
 * and a binary mask array (checked for != 0).
 */
CFFI_DLLEXPORT int grassfire(uint32_t* dst, const uint8_t* mask, int width, int height) {
    // 1st pass
    for (int x = 0; x < width; ++x) {
        for (int y = 0; y < height; ++y) {
            if (IMG_XY(mask, x, y) != 0) { // Pixel in contour
                // Get neighbors
                int north = (y == 0) ? 0 : IMG_XY(dst, x, y - 1);
                int west = (x == 0) ? 0 : IMG_XY(dst, x - 1, y);
                // Set value
                IMG_XY(dst, x, y) = 1 + min(north, west);
            }
        }
    }
    // 2nd pass
    for (int x = width - 1; x >= 0; x--) {
        for (int y = height - 1; y >= 0; y--) {
            if (IMG_XY(mask, x, y) != 0) { // Pixel in contour
                // Get neighbors
                uint32_t south = (y == (height - 1)) ?
                    0 : IMG_XY(dst, x, y + 1);
                uint32_t east = (x == (width - 1)) ?
                    0 : IMG_XY(dst, x + 1, y);
                // Set value
                IMG_XY(dst, x, y) = min(IMG_XY(dst, x, y),
                    1 + min(south, east));
            }
        }
    }
    return 0;
}
