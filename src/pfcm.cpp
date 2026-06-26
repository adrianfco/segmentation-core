#include "pfcm.hpp"

#include <algorithm>
#include <array>
#include <cmath>
#include <limits>
#include <random>
#include <stdexcept>
#include <vector>

namespace seg {

// PFCM combines fuzzy membership (u) and typicality (t) to update centroids:
//   centroid_i = sum_j (u_ij^m + a*t_ij^eta) * x_j
//              / sum_j (u_ij^m + a*t_ij^eta)
// where a balances fuzzy vs typicality influence (here fixed to 1.0).

namespace {

using Centroid = std::array<double, 3>;

double sq_dist(const unsigned char* px, const Centroid& c) {
    double dr = px[0] - c[0];
    double dg = px[1] - c[1];
    double db = px[2] - c[2];
    return dr * dr + dg * dg + db * db;
}

} // anonymous namespace

int pfcm_segment(Image& img, int k, int seed, int max_iters, double m, double eta) {
    if (k <= 0)          throw std::invalid_argument("k must be positive");
    if (max_iters <= 0)  throw std::invalid_argument("max_iters must be positive");
    if (m <= 1.0)        throw std::invalid_argument("m must be > 1");
    if (eta <= 1.0)      throw std::invalid_argument("eta must be > 1");

    const int n = img.width * img.height;
    if (n == 0) throw std::runtime_error("Image has no pixels");

    // Random centroid initialization (same approach as KMeans++ for reproducibility)
    std::mt19937 rng(static_cast<unsigned>(seed));
    std::vector<Centroid> centers(k);
    {
        std::uniform_int_distribution<int> pick(0, n - 1);
        int first = pick(rng);
        for (int c = 0; c < 3; ++c) centers[0][c] = img.data[first * 3 + c];

        std::vector<double> dist_sq(n);
        for (int ci = 1; ci < k; ++ci) {
            double total = 0.0;
            for (int p = 0; p < n; ++p) {
                dist_sq[p] = std::numeric_limits<double>::max();
                for (int prev = 0; prev < ci; ++prev) {
                    dist_sq[p] = std::min(dist_sq[p], sq_dist(&img.data[p * 3], centers[prev]));
                }
                total += dist_sq[p];
            }
            std::uniform_real_distribution<double> wheel(0.0, total);
            double r = wheel(rng), acc = 0.0;
            int chosen = n - 1;
            for (int p = 0; p < n; ++p) {
                acc += dist_sq[p];
                if (acc >= r) { chosen = p; break; }
            }
            for (int c = 0; c < 3; ++c) centers[ci][c] = img.data[chosen * 3 + c];
        }
    }

    const double exp_u   = 1.0 / (m   - 1.0);
    const double exp_t   = 1.0 / (eta - 1.0);
    const double a       = 1.0; // balance between membership and typicality

    std::vector<double> u(n * k, 1.0 / k);  // fuzzy membership matrix
    std::vector<double> t(n * k, 1.0 / k);  // typicality matrix

    // Per-cluster "gamma" for typicality normalization: sum_j u_ij^m * d_ij^2
    std::vector<double> gamma(k, 0.0);

    int iters = 0;
    for (; iters < max_iters; ++iters) {
        // --- Update centroids ---
        std::vector<Centroid> new_centers(k);
        std::vector<double>   denom(k, 0.0);
        for (auto& c : new_centers) c.fill(0.0);

        for (int p = 0; p < n; ++p) {
            const unsigned char* px = &img.data[p * 3];
            for (int i = 0; i < k; ++i) {
                double w = std::pow(u[p * k + i], m) + a * std::pow(t[p * k + i], eta);
                new_centers[i][0] += w * px[0];
                new_centers[i][1] += w * px[1];
                new_centers[i][2] += w * px[2];
                denom[i]          += w;
            }
        }
        double max_shift = 0.0;
        for (int i = 0; i < k; ++i) {
            if (denom[i] > 0) {
                double shift = 0.0;
                for (int c = 0; c < 3; ++c) {
                    double nv = new_centers[i][c] / denom[i];
                    shift += (nv - centers[i][c]) * (nv - centers[i][c]);
                    centers[i][c] = nv;
                }
                max_shift = std::max(max_shift, shift);
            }
        }

        // --- Update gamma (typicality scale per cluster) ---
        std::fill(gamma.begin(), gamma.end(), 0.0);
        std::vector<int> gamma_count(k, 0);
        for (int p = 0; p < n; ++p) {
            for (int i = 0; i < k; ++i) {
                gamma[i] += std::pow(u[p * k + i], m) * sq_dist(&img.data[p * 3], centers[i]);
                gamma_count[i]++;
            }
        }
        for (int i = 0; i < k; ++i) {
            if (gamma_count[i] > 0) gamma[i] /= gamma_count[i];
            if (gamma[i] < 1e-10)   gamma[i]  = 1e-10;
        }

        // --- Update membership u ---
        for (int p = 0; p < n; ++p) {
            const unsigned char* px = &img.data[p * 3];
            std::vector<double> dists(k);
            for (int i = 0; i < k; ++i) dists[i] = sq_dist(px, centers[i]);

            for (int i = 0; i < k; ++i) {
                if (dists[i] < 1e-10) {
                    // Exactly on center: full membership
                    for (int j = 0; j < k; ++j) u[p * k + j] = (i == j) ? 1.0 : 0.0;
                    break;
                }
                double sum = 0.0;
                for (int j = 0; j < k; ++j) {
                    if (dists[j] < 1e-10) { sum = std::numeric_limits<double>::infinity(); break; }
                    sum += std::pow(dists[i] / dists[j], exp_u);
                }
                u[p * k + i] = (sum > 0) ? 1.0 / sum : 0.0;
            }
        }

        // --- Update typicality t ---
        for (int p = 0; p < n; ++p) {
            const unsigned char* px = &img.data[p * 3];
            for (int i = 0; i < k; ++i) {
                double d = sq_dist(px, centers[i]);
                t[p * k + i] = 1.0 / (1.0 + std::pow(d / gamma[i], exp_t));
            }
        }

        if (max_shift < 1e-8) break;
    }

    // Assign each pixel to the cluster with highest membership
    for (int p = 0; p < n; ++p) {
        int best = 0;
        double best_u = u[p * k];
        for (int i = 1; i < k; ++i) {
            if (u[p * k + i] > best_u) { best_u = u[p * k + i]; best = i; }
        }
        const auto& c = centers[best];
        img.data[p * 3 + 0] = static_cast<unsigned char>(std::clamp(c[0], 0.0, 255.0));
        img.data[p * 3 + 1] = static_cast<unsigned char>(std::clamp(c[1], 0.0, 255.0));
        img.data[p * 3 + 2] = static_cast<unsigned char>(std::clamp(c[2], 0.0, 255.0));
    }

    return iters;
}

} // namespace seg
