#pragma once

#include "image_io.hpp"
#include "parallel.hpp"

#include <algorithm>
#include <array>
#include <limits>
#include <random>
#include <vector>

namespace seg::detail {

using Centroid = std::array<double, 3>;

inline double sq_dist(const unsigned char* px, const Centroid& c) {
    double dr = px[0] - c[0];
    double dg = px[1] - c[1];
    double db = px[2] - c[2];
    return dr * dr + dg * dg + db * db;
}

// KMeans++ seeding, shared by both algorithms: the first centroid is uniform,
// every later one is drawn with probability proportional to the pixel's squared
// distance to the closest centroid picked so far. dist_sq holds that closest
// distance and survives between rounds, so a round only measures against the
// centroid just added and seeding stays O(n*k).
inline std::vector<Centroid> kmeans_plus_plus(const Image& img, int k, std::mt19937& rng) {
    const int n = img.width * img.height;

    std::vector<Centroid> centers(k);
    std::uniform_int_distribution<int> pick(0, n - 1);
    int first = pick(rng);
    for (int c = 0; c < 3; ++c) centers[0][c] = img.data[first * 3 + c];
    if (k == 1) return centers;

    std::vector<double> dist_sq(n, std::numeric_limits<double>::max());
    for (int ci = 1; ci < k; ++ci) {
        SEG_OMP(parallel for schedule(static))
        for (int p = 0; p < n; ++p) {
            dist_sq[p] = std::min(dist_sq[p], sq_dist(&img.data[p * 3], centers[ci - 1]));
        }
        // summed serially so the picked centroid does not depend on threads
        double total = 0.0;
        for (int p = 0; p < n; ++p) total += dist_sq[p];

        std::uniform_real_distribution<double> wheel(0.0, total);
        double r = wheel(rng), acc = 0.0;
        int chosen = n - 1;
        for (int p = 0; p < n; ++p) {
            acc += dist_sq[p];
            if (acc >= r) { chosen = p; break; }
        }
        for (int c = 0; c < 3; ++c) centers[ci][c] = img.data[chosen * 3 + c];
    }
    return centers;
}

} // namespace seg::detail
