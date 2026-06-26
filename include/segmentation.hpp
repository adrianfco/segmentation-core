#pragma once

#include <string>

namespace seg {

enum class Algorithm {
    KMeans,
    PFCM,
};

struct SegmentationParams {
    std::string input_path;
    std::string output_path;
    Algorithm   algorithm  = Algorithm::KMeans;
    int         k          = 4;
    int         seed       = 42;
    int         max_iters  = 100;

    // PFCM-specific
    double pfcm_m   = 2.0;  // fuzziness exponent
    double pfcm_eta = 2.0;  // typicality exponent
};

struct SegmentationResult {
    bool        success       = false;
    std::string output_path;
    double      runtime_ms    = 0.0;
    int         iterations    = 0;
    int         width         = 0;
    int         height        = 0;
    std::string algorithm;
    std::string error_message;
};

SegmentationResult segment_image(const SegmentationParams& params);

} // namespace seg
