#pragma once

#include "image_io.hpp"

namespace seg {

// Segments img in place and returns the iteration count.
int kmeans_segment(Image& img, int k, int seed, int max_iters);

} // namespace seg
