#pragma once

#include "image_io.hpp"

namespace seg {

// Segments img in place and returns the iteration count. m is the fuzziness
// exponent and eta the typicality exponent, both must be > 1.
int pfcm_segment(Image& img, int k, int seed, int max_iters,
                 double m = 2.0, double eta = 2.0);

} // namespace seg
