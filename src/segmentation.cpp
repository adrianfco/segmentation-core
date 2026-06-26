#include "segmentation.hpp"
#include "image_io.hpp"
#include "kmeans.hpp"
#include "pfcm.hpp"

#include <chrono>
#include <stdexcept>

namespace seg {

static std::string algorithm_name(Algorithm alg) {
    switch (alg) {
        case Algorithm::KMeans: return "kmeans";
        case Algorithm::PFCM:   return "pfcm";
    }
    return "unknown";
}

SegmentationResult segment_image(const SegmentationParams& params) {
    SegmentationResult result;
    result.algorithm = algorithm_name(params.algorithm);

    // Parameter validation
    if (params.input_path.empty()) {
        result.error_message = "input_path is empty";
        return result;
    }
    if (params.output_path.empty()) {
        result.error_message = "output_path is empty";
        return result;
    }
    if (params.k < 1) {
        result.error_message = "k must be >= 1";
        return result;
    }
    if (params.max_iters < 1) {
        result.error_message = "max_iters must be >= 1";
        return result;
    }

    auto t_start = std::chrono::high_resolution_clock::now();

    try {
        Image img = load_image(params.input_path);
        result.width  = img.width;
        result.height = img.height;

        int iters = 0;
        switch (params.algorithm) {
            case Algorithm::KMeans:
                iters = kmeans_segment(img, params.k, params.seed, params.max_iters);
                break;
            case Algorithm::PFCM:
                iters = pfcm_segment(img, params.k, params.seed, params.max_iters,
                                     params.pfcm_m, params.pfcm_eta);
                break;
        }
        result.iterations = iters;

        save_image(params.output_path, img);
        result.output_path = params.output_path;

    } catch (const std::exception& e) {
        result.error_message = e.what();
        auto t_end = std::chrono::high_resolution_clock::now();
        result.runtime_ms = std::chrono::duration<double, std::milli>(t_end - t_start).count();
        return result;
    }

    auto t_end = std::chrono::high_resolution_clock::now();
    result.runtime_ms = std::chrono::duration<double, std::milli>(t_end - t_start).count();
    result.success = true;
    return result;
}

} // namespace seg
