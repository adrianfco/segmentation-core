#include "kmeans.hpp"

#include <algorithm>
#include <array>
#include <cmath>
#include <limits>
#include <random>
#include <stdexcept>
#include <vector>

namespace seg {

namespace {

using Centroid = std::array<double, 3>;

int nearest_centroid(const unsigned char* pixel, const std::vector<Centroid>& centroids) {
    double best = std::numeric_limits<double>::max();
    int    idx  = 0;
    for (int i = 0; i < static_cast<int>(centroids.size()); ++i) {
        double dr = pixel[0] - centroids[i][0];
        double dg = pixel[1] - centroids[i][1];
        double db = pixel[2] - centroids[i][2];
        double d  = dr * dr + dg * dg + db * db;
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

    // KMeans++ initialization for better convergence
    std::mt19937 rng(static_cast<unsigned>(seed));

    std::vector<Centroid> centroids(k);
    {
        std::uniform_int_distribution<int> pick(0, n_pixels - 1);
        int first = pick(rng);
        for (int c = 0; c < 3; ++c) centroids[0][c] = img.data[first * 3 + c];

        std::vector<double> dist_sq(n_pixels);
        for (int ci = 1; ci < k; ++ci) {
            double total = 0.0;
            for (int p = 0; p < n_pixels; ++p) {
                dist_sq[p] = std::numeric_limits<double>::max();
                for (int prev = 0; prev < ci; ++prev) {
                    double dr = img.data[p * 3 + 0] - centroids[prev][0];
                    double dg = img.data[p * 3 + 1] - centroids[prev][1];
                    double db = img.data[p * 3 + 2] - centroids[prev][2];
                    double d  = dr * dr + dg * dg + db * db;
                    dist_sq[p] = std::min(dist_sq[p], d);
                }
                total += dist_sq[p];
            }
            std::uniform_real_distribution<double> wheel(0.0, total);
            double r = wheel(rng);
            double acc = 0.0;
            int chosen = n_pixels - 1;
            for (int p = 0; p < n_pixels; ++p) {
                acc += dist_sq[p];
                if (acc >= r) { chosen = p; break; }
            }
            for (int c = 0; c < 3; ++c) centroids[ci][c] = img.data[chosen * 3 + c];
        }
    }

    std::vector<int>     labels(n_pixels);
    std::vector<Centroid> sums(k);
    std::vector<int>     counts(k);

    int iters = 0;
    for (; iters < max_iters; ++iters) {
        bool changed = false;

        // Assignment
        for (int p = 0; p < n_pixels; ++p) {
            int lbl = nearest_centroid(&img.data[p * 3], centroids);
            if (lbl != labels[p]) { labels[p] = lbl; changed = true; }
        }

        if (!changed && iters > 0) break;

        // Update
        for (auto& s : sums)   s.fill(0.0);
        std::fill(counts.begin(), counts.end(), 0);
        for (int p = 0; p < n_pixels; ++p) {
            int lbl = labels[p];
            sums[lbl][0] += img.data[p * 3 + 0];
            sums[lbl][1] += img.data[p * 3 + 1];
            sums[lbl][2] += img.data[p * 3 + 2];
            counts[lbl]++;
        }
        for (int i = 0; i < k; ++i) {
            if (counts[i] > 0) {
                centroids[i][0] = sums[i][0] / counts[i];
                centroids[i][1] = sums[i][1] / counts[i];
                centroids[i][2] = sums[i][2] / counts[i];
            }
        }
    }

    // Replace each pixel with its centroid color
    for (int p = 0; p < n_pixels; ++p) {
        const auto& c = centroids[labels[p]];
        img.data[p * 3 + 0] = static_cast<unsigned char>(std::clamp(c[0], 0.0, 255.0));
        img.data[p * 3 + 1] = static_cast<unsigned char>(std::clamp(c[1], 0.0, 255.0));
        img.data[p * 3 + 2] = static_cast<unsigned char>(std::clamp(c[2], 0.0, 255.0));
    }

    return iters;
}

} // namespace seg
