#include "kmeans.hpp"

#include "centroids.hpp"
#include "parallel.hpp"

#include <algorithm>
#include <array>
#include <cmath>
#include <limits>
#include <random>
#include <stdexcept>
#include <vector>

namespace seg {

namespace {

using detail::Centroid;
using detail::sq_dist;

using Sum = std::array<long long, 3>;

int nearest_centroid(const unsigned char* pixel, const std::vector<Centroid>& centroids) {
    double best = std::numeric_limits<double>::max();
    int    idx  = 0;
    for (int i = 0; i < static_cast<int>(centroids.size()); ++i) {
        double d = sq_dist(pixel, centroids[i]);
        if (d < best) {
            best = d;
            idx  = i;
        }
    }
    return idx;
}

} // anonymous namespace

int kmeans_segment(Image& img, int k, int seed, int max_iters) {
    if (k <= 0) throw std::invalid_argument("k must be positive");
    if (max_iters <= 0) throw std::invalid_argument("max_iters must be positive");

    const int n_pixels = img.width * img.height;
    if (n_pixels == 0) throw std::runtime_error("Image has no pixels");

    std::mt19937 rng(static_cast<unsigned>(seed));
    std::vector<Centroid> centroids = detail::kmeans_plus_plus(img, k, rng);

    std::vector<int>  labels(n_pixels);
    std::vector<Sum>  sums(k);
    std::vector<int>  counts(k);

    int iters = 0;
    for (; iters < max_iters; ++iters) {
        bool changed = false;

        // Assignment
        SEG_OMP(parallel for schedule(static) reduction(||:changed))
        for (int p = 0; p < n_pixels; ++p) {
            int lbl = nearest_centroid(&img.data[p * 3], centroids);
            if (lbl != labels[p]) { labels[p] = lbl; changed = true; }
        }

        if (!changed && iters > 0) break;

        // Update. The sums are integers, so merging the per-thread partials is
        // exact whatever order they arrive in.
        for (auto& s : sums) s.fill(0);
        std::fill(counts.begin(), counts.end(), 0);

        SEG_OMP(parallel)
        {
            std::vector<Sum> local_sums(k);
            std::vector<int> local_counts(k);

            SEG_OMP(for schedule(static) nowait)
            for (int p = 0; p < n_pixels; ++p) {
                int lbl = labels[p];
                local_sums[lbl][0] += img.data[p * 3 + 0];
                local_sums[lbl][1] += img.data[p * 3 + 1];
                local_sums[lbl][2] += img.data[p * 3 + 2];
                local_counts[lbl]++;
            }

            SEG_OMP(critical)
            for (int i = 0; i < k; ++i) {
                for (int c = 0; c < 3; ++c) sums[i][c] += local_sums[i][c];
                counts[i] += local_counts[i];
            }
        }

        for (int i = 0; i < k; ++i) {
            if (counts[i] > 0) {
                for (int c = 0; c < 3; ++c) {
                    centroids[i][c] = static_cast<double>(sums[i][c]) / counts[i];
                }
            }
        }
    }

    // Replace each pixel with its centroid color
    SEG_OMP(parallel for schedule(static))
    for (int p = 0; p < n_pixels; ++p) {
        const auto& c = centroids[labels[p]];
        img.data[p * 3 + 0] = static_cast<unsigned char>(std::clamp(c[0], 0.0, 255.0));
        img.data[p * 3 + 1] = static_cast<unsigned char>(std::clamp(c[1], 0.0, 255.0));
        img.data[p * 3 + 2] = static_cast<unsigned char>(std::clamp(c[2], 0.0, 255.0));
    }

    return iters;
}

} // namespace seg
