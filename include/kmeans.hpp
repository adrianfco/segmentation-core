#pragma once

#include "image_io.hpp"

namespace seg {

// Runs KMeans color segmentation in-place on img.
// Returns the number of iterations executed.
// seed controls centroid initialization; result is reproducible for fixed seed.
int kmeans_segment(Image& img, int k, int seed, int max_iters);

} // namespace seg
