#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

#include "segmentation.hpp"

namespace py = pybind11;

namespace {

seg::Algorithm parse_algorithm(const std::string& name) {
    if (name == "kmeans") return seg::Algorithm::KMeans;
    if (name == "pfcm")   return seg::Algorithm::PFCM;
    throw std::invalid_argument("Unknown algorithm: '" + name +
                                "'. Supported: kmeans, pfcm.");
}

} // anonymous namespace

PYBIND11_MODULE(segmentation_core, m) {
    m.doc() = "CPU image segmentation engine — KMeans and PFCM";

    py::class_<seg::SegmentationResult>(m, "SegmentationResult")
        .def_readonly("success",       &seg::SegmentationResult::success)
        .def_readonly("output_path",   &seg::SegmentationResult::output_path)
        .def_readonly("runtime_ms",    &seg::SegmentationResult::runtime_ms)
        .def_readonly("iterations",    &seg::SegmentationResult::iterations)
        .def_readonly("width",         &seg::SegmentationResult::width)
        .def_readonly("height",        &seg::SegmentationResult::height)
        .def_readonly("algorithm",     &seg::SegmentationResult::algorithm)
        .def_readonly("error_message", &seg::SegmentationResult::error_message)
        .def("__repr__", [](const seg::SegmentationResult& r) {
            return "<SegmentationResult success=" + std::string(r.success ? "True" : "False") +
                   " algorithm=" + r.algorithm +
                   " runtime_ms=" + std::to_string(r.runtime_ms) + ">";
        });

    m.def("segment_image",
        [](const std::string& image_path,
           const std::string& output_path,
           const std::string& algorithm,
           int   k,
           int   seed,
           int   max_iters,
           double pfcm_m,
           double pfcm_eta) -> seg::SegmentationResult
        {
            seg::SegmentationParams p;
            p.input_path  = image_path;
            p.output_path = output_path;
            p.algorithm   = parse_algorithm(algorithm);
            p.k           = k;
            p.seed        = seed;
            p.max_iters   = max_iters;
            p.pfcm_m      = pfcm_m;
            p.pfcm_eta    = pfcm_eta;
            return seg::segment_image(p);
        },
        py::arg("image_path"),
        py::arg("output_path"),
        py::arg("algorithm") = "kmeans",
        py::arg("k")         = 4,
        py::arg("seed")      = 42,
        py::arg("max_iters") = 100,
        py::arg("pfcm_m")    = 2.0,
        py::arg("pfcm_eta")  = 2.0,
        R"(Segment an image using the specified algorithm.

Args:
    image_path:  Path to the input image (PNG, JPG, BMP).
    output_path: Path where the segmented image will be written.
    algorithm:   One of 'kmeans' or 'pfcm'. Default: 'kmeans'.
    k:           Number of segments/clusters. Default: 4.
    seed:        Random seed for reproducibility. Default: 42.
    max_iters:   Maximum number of iterations. Default: 100.
    pfcm_m:      PFCM fuzziness exponent (>1). Default: 2.0.
    pfcm_eta:    PFCM typicality exponent (>1). Default: 2.0.

Returns:
    SegmentationResult with success flag, runtime, and metadata.
)"
    );
}
