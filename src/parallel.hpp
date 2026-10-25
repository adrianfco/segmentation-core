#pragma once

#include <algorithm>

namespace seg::detail {

// Float sums go per fixed-size chunk and the partials are merged in chunk
// order, so the result is bit-identical for any thread count and without OpenMP.
constexpr int kChunkPixels = 65536;

inline int chunk_count(int n_pixels) {
    return (n_pixels + kChunkPixels - 1) / kChunkPixels;
}

inline int chunk_end(int chunk, int n_pixels) {
    return std::min((chunk + 1) * kChunkPixels, n_pixels);
}

// Rounded up to a cache line (8 doubles) so neighbouring chunks don't false-share.
inline int partial_stride(int values_per_chunk) {
    return (values_per_chunk + 7) / 8 * 8;
}

} // namespace seg::detail
