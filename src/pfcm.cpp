#include "pfcm.hpp"

#include "centroids.hpp"

#include <algorithm>
#include <cmath>
#include <limits>
#include <random>
#include <stdexcept>
#include <vector>

namespace seg {

namespace {

using detail::Centroid;
using detail::sq_dist;

// Float sums are computed on chunks to preserve deterministic output on parallelization
constexpr int kChunkPixels = 65536;

int chunk_count(int n_pixels) {
    return (n_pixels + kChunkPixels - 1) / kChunkPixels;
}

int chunk_end(int chunk, int n_pixels) {
    return std::min((chunk + 1) * kChunkPixels, n_pixels);
}

// Round up to a cache line (8 doubles) so neighbouring chunks don't false share
int partial_stride(int values_per_chunk) {
    return (values_per_chunk + 7) / 8 * 8;
}

// Fast path for default values
double fast_pow(double x, double e) {
    if (e == 1.0) return x;
    if (e == 2.0) return x * x;
    return std::pow(x, e);
}

} // anonymous namespace

int pfcm_segment(Image& img, int k, int seed, int max_iters, double m, double eta) {
    if (k <= 0)          throw std::invalid_argument("k must be positive");
    if (max_iters <= 0)  throw std::invalid_argument("max_iters must be positive");
    if (m <= 1.0)        throw std::invalid_argument("m must be > 1");
    if (eta <= 1.0)      throw std::invalid_argument("eta must be > 1");

    const int n = img.width * img.height;
    if (n == 0) throw std::runtime_error("Image has no pixels");

    std::mt19937 rng(static_cast<unsigned>(seed));
    std::vector<Centroid> centers = detail::kmeans_plus_plus(img, k, rng);

    const double exp_u = 1.0 / (m   - 1.0);
    const double exp_t = 1.0 / (eta - 1.0);
    const double a     = 1.0; // balance between membership and typicality

    std::vector<double> u(n * k);
    std::vector<double> t(n * k);
    std::vector<double> gamma(k, 0.0);

    const int n_chunks      = chunk_count(n);
    const int gamma_stride  = partial_stride(2 * k);  // num, den per cluster
    const int center_stride = partial_stride(4 * k);  // r, g, b, weight

    // u and t are computed before the first update, otherwise the uniform
    // u = 1/k collapses every centroid to the image mean.
    auto update_u = [&]() {
        #pragma omp parallel
        {
            std::vector<double> dists(k);
            std::vector<double> w(k);

            #pragma omp for schedule(static)
            for (int p = 0; p < n; ++p) {
                const unsigned char* px = &img.data[p * 3];
                int    nearest = 0;
                double d_min   = std::numeric_limits<double>::max();
                for (int i = 0; i < k; ++i) {
                    dists[i] = sq_dist(px, centers[i]);
                    if (dists[i] < d_min) { d_min = dists[i]; nearest = i; }
                }

                if (d_min < 1e-10) {
                    for (int j = 0; j < k; ++j) u[p * k + j] = (nearest == j) ? 1.0 : 0.0;
                    continue;
                }

                // u_i = d_i^-e / sum_j d_j^-e with d the squared distance and
                // e = 1/(m-1). Scaled by d_min so the nearest term is 1.
                double sum = 0.0;
                for (int i = 0; i < k; ++i) {
                    w[i] = fast_pow(d_min / dists[i], exp_u);
                    sum += w[i];
                }
                for (int i = 0; i < k; ++i) u[p * k + i] = w[i] / sum;
            }
        }
    };

    auto update_gamma_and_t = [&]() {
        // gamma_i = sum_p u^m d / sum_p u^m
        std::vector<double> partials(static_cast<size_t>(n_chunks) * gamma_stride, 0.0);

        #pragma omp parallel for schedule(static)
        for (int c = 0; c < n_chunks; ++c) {
            double* acc = &partials[static_cast<size_t>(c) * gamma_stride];
            for (int p = c * kChunkPixels; p < chunk_end(c, n); ++p) {
                for (int i = 0; i < k; ++i) {
                    double um = fast_pow(u[p * k + i], m);
                    acc[i * 2 + 0] += um * sq_dist(&img.data[p * 3], centers[i]);
                    acc[i * 2 + 1] += um;
                }
            }
        }

        std::vector<double> num(k, 0.0), den(k, 0.0);
        for (int c = 0; c < n_chunks; ++c) {
            const double* acc = &partials[static_cast<size_t>(c) * gamma_stride];
            for (int i = 0; i < k; ++i) {
                num[i] += acc[i * 2 + 0];
                den[i] += acc[i * 2 + 1];
            }
        }

        for (int i = 0; i < k; ++i) {
            gamma[i] = (den[i] > 0) ? num[i] / den[i] : 1e-10;
            if (gamma[i] < 1e-10) gamma[i] = 1e-10;
        }

        // one reciprocal per cluster instead of a divide per pixel and cluster
        std::vector<double> inv_gamma(k);
        for (int i = 0; i < k; ++i) inv_gamma[i] = 1.0 / gamma[i];

        #pragma omp parallel for schedule(static)
        for (int p = 0; p < n; ++p) {
            const unsigned char* px = &img.data[p * 3];
            for (int i = 0; i < k; ++i) {
                double d = sq_dist(px, centers[i]);
                t[p * k + i] = 1.0 / (1.0 + fast_pow(d * inv_gamma[i], exp_t));
            }
        }
    };

    update_u();
    update_gamma_and_t();

    int iters = 0;
    for (; iters < max_iters; ++iters) {
        std::vector<Centroid> new_centers(k);
        std::vector<double>   denom(k, 0.0);
        for (auto& c : new_centers) c.fill(0.0);

        // centroid_i = sum_p w x_p / sum_p w, with w = u^m + a*t^eta
        std::vector<double> partials(static_cast<size_t>(n_chunks) * center_stride, 0.0);

        #pragma omp parallel for schedule(static)
        for (int c = 0; c < n_chunks; ++c) {
            double* acc = &partials[static_cast<size_t>(c) * center_stride];
            for (int p = c * kChunkPixels; p < chunk_end(c, n); ++p) {
                const unsigned char* px = &img.data[p * 3];
                for (int i = 0; i < k; ++i) {
                    double w = fast_pow(u[p * k + i], m) + a * fast_pow(t[p * k + i], eta);
                    acc[i * 4 + 0] += w * px[0];
                    acc[i * 4 + 1] += w * px[1];
                    acc[i * 4 + 2] += w * px[2];
                    acc[i * 4 + 3] += w;
                }
            }
        }

        for (int c = 0; c < n_chunks; ++c) {
            const double* acc = &partials[static_cast<size_t>(c) * center_stride];
            for (int i = 0; i < k; ++i) {
                for (int ch = 0; ch < 3; ++ch) new_centers[i][ch] += acc[i * 4 + ch];
                denom[i] += acc[i * 4 + 3];
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

        if (max_shift < 1e-8) break;

        update_u();
        update_gamma_and_t();
    }

    #pragma omp parallel for schedule(static)
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
