#pragma once

#include <algorithm>

// SEG_OMP(parallel for schedule(static)) expands to the matching #pragma when
// the build has OpenMP, and to nothing otherwise, so a serial build stays free
// of -Wunknown-pragmas warnings.
#ifdef _OPENMP
#  define SEG_OMP_STR(x) #x
#  define SEG_OMP(directive) _Pragma(SEG_OMP_STR(omp directive))
#else
#  define SEG_OMP(directive)
#endif

namespace seg::detail {

// Floating point sums over pixels are accumulated per fixed-size chunk and the
// chunk partials are merged serially in chunk order. Chunk boundaries do not
// depend on how OpenMP splits the loop, so the result is bit-identical for any
// number of threads and for a build without OpenMP.
constexpr int kChunkPixels = 65536;

inline int chunk_count(int n_pixels) {
    return (n_pixels + kChunkPixels - 1) / kChunkPixels;
}

inline int chunk_end(int chunk, int n_pixels) {
    return std::min((chunk + 1) * kChunkPixels, n_pixels);
}

// Doubles per chunk in a partials buffer, rounded up to a 64 byte cache line so
// two threads never write to the same line.
inline int partial_stride(int values_per_chunk) {
    return (values_per_chunk + 7) / 8 * 8;
}

} // namespace seg::detail
