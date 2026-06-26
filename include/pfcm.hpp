#pragma once

#include "image_io.hpp"

namespace seg {

// Runs PFCM color segmentation in-place on img.
// m:   fuzziness exponent (>1, typically 2.0)
// eta: typicality exponent (>1, typically 2.0)
// Returns the number of iterations executed.
int pfcm_segment(Image& img, int k, int seed, int max_iters,
                 double m = 2.0, double eta = 2.0);

} // namespace seg
