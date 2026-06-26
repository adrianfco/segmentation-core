#include "segmentation.hpp"
#include "image_io.hpp"
#include "kmeans.hpp"
#include "pfcm.hpp"

#include <cassert>
#include <cstdlib>
#include <filesystem>
#include <iostream>
#include <string>

// Minimal assertion helper that prints context on failure.
#define CHECK(cond)                                                          \
    do {                                                                     \
        if (!(cond)) {                                                       \
            std::cerr << "FAIL: " #cond "\n  at " __FILE__ ":"              \
                      << __LINE__ << "\n";                                   \
            std::exit(1);                                                    \
        }                                                                    \
    } while (false)

namespace fs = std::filesystem;

// ----------------------------------------------------------------
// Helpers
// ----------------------------------------------------------------

static seg::Image make_test_image(int w = 16, int h = 16) {
    seg::Image img;
    img.width    = w;
    img.height   = h;
    img.channels = 3;
    img.data.resize(w * h * 3);
    // Four-quadrant synthetic image with distinct colors
    for (int y = 0; y < h; ++y) {
        for (int x = 0; x < w; ++x) {
            int p = (y * w + x) * 3;
            if (x < w / 2 && y < h / 2) {
                img.data[p] = 255; img.data[p+1] = 0;   img.data[p+2] = 0;   // red
            } else if (x >= w / 2 && y < h / 2) {
                img.data[p] = 0;   img.data[p+1] = 255; img.data[p+2] = 0;   // green
            } else if (x < w / 2 && y >= h / 2) {
                img.data[p] = 0;   img.data[p+1] = 0;   img.data[p+2] = 255; // blue
            } else {
                img.data[p] = 255; img.data[p+1] = 255; img.data[p+2] = 0;   // yellow
            }
        }
    }
    return img;
}

// ----------------------------------------------------------------
// Tests
// ----------------------------------------------------------------

static void test_params_validation() {
    // Empty input path
    seg::SegmentationParams p;
    p.input_path  = "";
    p.output_path = "/tmp/out.png";
    auto r = seg::segment_image(p);
    CHECK(!r.success);
    CHECK(!r.error_message.empty());

    // Empty output path
    p.input_path  = "some_file.png";
    p.output_path = "";
    r = seg::segment_image(p);
    CHECK(!r.success);

    // Invalid k
    p.output_path = "/tmp/out.png";
    p.k = 0;
    r = seg::segment_image(p);
    CHECK(!r.success);

    // Nonexistent input file
    p.k          = 4;
    p.input_path = "/nonexistent/path/image.png";
    r = seg::segment_image(p);
    CHECK(!r.success);
    CHECK(!r.error_message.empty());
}

static void test_kmeans_basic() {
    seg::Image img = make_test_image();
    int iters = seg::kmeans_segment(img, 4, 42, 100);
    CHECK(iters >= 1);
    CHECK(!img.data.empty());
}

static void test_kmeans_reproducible() {
    seg::Image a = make_test_image();
    seg::Image b = make_test_image();
    seg::kmeans_segment(a, 4, 42, 100);
    seg::kmeans_segment(b, 4, 42, 100);
    CHECK(a.data == b.data);
}

static void test_kmeans_different_seeds() {
    seg::Image a = make_test_image();
    seg::Image b = make_test_image();
    seg::kmeans_segment(a, 4, 1, 100);
    seg::kmeans_segment(b, 4, 999, 100);
    // Different seeds are very likely to produce different results on this image.
    // This test just ensures both runs complete and produce valid pixel data.
    CHECK(!a.data.empty());
    CHECK(!b.data.empty());
}

static void test_pfcm_basic() {
    seg::Image img = make_test_image();
    int iters = seg::pfcm_segment(img, 4, 42, 50);
    CHECK(iters >= 1);
    CHECK(!img.data.empty());
}

static void test_pfcm_reproducible() {
    seg::Image a = make_test_image();
    seg::Image b = make_test_image();
    seg::pfcm_segment(a, 4, 42, 50);
    seg::pfcm_segment(b, 4, 42, 50);
    CHECK(a.data == b.data);
}

static void test_segment_image_kmeans_end_to_end() {
    // Write the synthetic image to a temp file then run via the public API.
    const std::string tmp_in  = "/tmp/seg_test_input.png";
    const std::string tmp_out = "/tmp/seg_test_output_kmeans.png";

    seg::Image img = make_test_image(32, 32);
    seg::save_image(tmp_in, img);

    seg::SegmentationParams p;
    p.input_path  = tmp_in;
    p.output_path = tmp_out;
    p.algorithm   = seg::Algorithm::KMeans;
    p.k           = 4;
    p.seed        = 42;
    p.max_iters   = 100;

    auto r = seg::segment_image(p);
    CHECK(r.success);
    CHECK(r.width  == 32);
    CHECK(r.height == 32);
    CHECK(r.runtime_ms >= 0.0);
    CHECK(r.iterations >= 1);
    CHECK(r.algorithm == "kmeans");
    CHECK(fs::exists(tmp_out));
}

static void test_segment_image_pfcm_end_to_end() {
    const std::string tmp_in  = "/tmp/seg_test_input.png";
    const std::string tmp_out = "/tmp/seg_test_output_pfcm.png";

    seg::Image img = make_test_image(16, 16);
    seg::save_image(tmp_in, img);

    seg::SegmentationParams p;
    p.input_path  = tmp_in;
    p.output_path = tmp_out;
    p.algorithm   = seg::Algorithm::PFCM;
    p.k           = 4;
    p.seed        = 42;
    p.max_iters   = 30;

    auto r = seg::segment_image(p);
    CHECK(r.success);
    CHECK(r.algorithm == "pfcm");
    CHECK(fs::exists(tmp_out));
}

static void test_image_io_roundtrip() {
    const std::string path = "/tmp/seg_io_roundtrip.png";
    seg::Image orig = make_test_image(8, 8);
    seg::save_image(path, orig);

    seg::Image loaded = seg::load_image(path);
    CHECK(loaded.width    == orig.width);
    CHECK(loaded.height   == orig.height);
    CHECK(loaded.channels == orig.channels);
    CHECK(loaded.data     == orig.data);
}

// ----------------------------------------------------------------
// Runner
// ----------------------------------------------------------------

int main() {
    std::cout << "Running segmentation tests...\n";

    test_params_validation();            std::cout << "  params_validation       OK\n";
    test_kmeans_basic();                 std::cout << "  kmeans_basic            OK\n";
    test_kmeans_reproducible();          std::cout << "  kmeans_reproducible     OK\n";
    test_kmeans_different_seeds();       std::cout << "  kmeans_different_seeds  OK\n";
    test_pfcm_basic();                   std::cout << "  pfcm_basic              OK\n";
    test_pfcm_reproducible();            std::cout << "  pfcm_reproducible       OK\n";
    test_segment_image_kmeans_end_to_end(); std::cout << "  segment_image_kmeans    OK\n";
    test_segment_image_pfcm_end_to_end();   std::cout << "  segment_image_pfcm      OK\n";
    test_image_io_roundtrip();           std::cout << "  image_io_roundtrip      OK\n";

    std::cout << "\nAll tests passed.\n";
    return 0;
}
