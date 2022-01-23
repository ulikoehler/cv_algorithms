#pragma once

/** 
 * Dirty macro to directly access an image at X/Y coordinates.
 * Assumes that the width variable is defined to the width of the image
 */
#define IMG_XY(img, x, y) img[(x) + (width)*(y)]

// Windows compatibility
#ifndef CFFI_DLLEXPORT
#if defined(_MSC_VER)
#define CFFI_DLLEXPORT __declspec(dllexport)
#else // !defined(_MSC_VER)
#define CFFI_DLLEXPORT
#endif // defined(_MSC_VER)
#endif // ifndef CFFI_DLLEXPORT
