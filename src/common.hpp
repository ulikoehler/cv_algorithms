#pragma once

/** 
 * Dirty macro to directly access an image at X/Y coordinates.
 * Assumes that the width variable is defined to the width of the image
 */
#define IMG_XY(img, x, y) img[(x) + (width)*(y)]
