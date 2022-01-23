#include "common.hpp"
#include <stdlib.h>
#include <assert.h>
#include <stdbool.h>
#include <limits.h>
#include <string.h>
#include <stdint.h>

#ifdef _MSC_VER
#include <intrin.h>
#define __builtin_popcount __popcnt
#define __builtin_popcountll __popcnt64
#endif


//Forward declaration required due to CFFI's requirement to have unmangled symbols
extern "C" {
    CFFI_DLLEXPORT int popcount8(uint8_t* dst, const uint8_t* src, int size);
    CFFI_DLLEXPORT int popcount16(uint8_t* dst, const uint16_t* src, int size);
    CFFI_DLLEXPORT int popcount32(uint8_t* dst, const uint32_t* src, int size);
    CFFI_DLLEXPORT int popcount64(uint8_t* dst, const uint64_t* src, int size);
}

CFFI_DLLEXPORT int popcount8(uint8_t* dst, const uint8_t* src, int size) {
    for (int i = 0; i < size; ++i) {
        dst[i] = __builtin_popcount(src[i]);
    }
    return 0;
}

CFFI_DLLEXPORT int popcount16(uint8_t* dst, const uint16_t* src, int size) {
    for (int i = 0; i < size; ++i) {
        dst[i] = __builtin_popcount(src[i]);
    }
    return 0;
}

CFFI_DLLEXPORT int popcount32(uint8_t* dst, const uint32_t* src, int size) {
    for (int i = 0; i < size; ++i) {
        dst[i] = __builtin_popcount(src[i]);
    }
    return 0;
}

CFFI_DLLEXPORT int popcount64(uint8_t* dst, const uint64_t* src, int size) {
    for (int i = 0; i < size; ++i) {
        dst[i] = __builtin_popcountll(src[i]);
    }
    return 0;
}
