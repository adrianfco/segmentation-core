#include "image_io.hpp"
#include "kmeans.hpp"
#include "pfcm.hpp"

#include <algorithm>
#include <array>
#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <random>
#include <sstream>
#include <stdexcept>
#include <string>
#include <vector>

// Times kmeans_segment / pfcm_segment on in-memory synthetic images and writes
// one CSV row per config to stdout. Image I/O is excluded so the numbers
// reflect the algorithms only.
//
//   ./bench_segmentation > baseline.csv
//   ./bench_segmentation --algo pfcm --mp 16 | column -t -s,

namespace {

constexpr int    kSeed         = 42;
constexpr int    kIters        = 10;      // max_iters passed to the algorithms
constexpr int    kMaxReps      = 3;
constexpr double kBudgetMs     = 10000.0; // stop repeating a config after this much time
constexpr int    kTrueClusters = 8;

struct Options {
    std::vector<std::string> algos = {"kmeans", "pfcm"};
    std::vector<int>         mps   = {1, 4, 16};
    std::vector<int>         ks    = {4, 8, 16};
};

struct Timing {
    int    iters;
    int    reps;
    double median_ms;
    double min_ms;
};

[[noreturn]] void usage(const char* prog) {
    std::fprintf(stderr, "usage: %s [--algo kmeans|pfcm|all] [--mp 1,4,16] [--k 4,8,16]\n", prog);
    std::exit(2);
}

std::vector<int> parse_int_list(const std::string& s) {
    std::vector<int> out;
    std::stringstream ss(s);
    std::string item;
    while (std::getline(ss, item, ',')) {
        int v = std::stoi(item);
        if (v < 1) throw std::invalid_argument("values must be positive");
        out.push_back(v);
    }
    return out;
}

Options parse_args(int argc, char** argv) {
    Options opt;
    for (int i = 1; i < argc; ++i) {
        std::string arg = argv[i];
        if (i + 1 >= argc) usage(argv[0]);
        std::string value = argv[++i];
        try {
            if (arg == "--algo") {
                if (value == "all")                                opt.algos = {"kmeans", "pfcm"};
                else if (value == "kmeans" || value == "pfcm")     opt.algos = {value};
                else usage(argv[0]);
            }
            else if (arg == "--mp") opt.mps = parse_int_list(value);
            else if (arg == "--k")  opt.ks  = parse_int_list(value);
            else usage(argv[0]);
        } catch (const std::logic_error&) {
            usage(argv[0]);
        }
    }
    return opt;
}

int side_for_mp(int mp) {
    int side = 1024;
    while (side * side < mp * 1024 * 1024) side *= 2;
    return side;
}

// Pixels drawn from kTrueClusters random colors plus Gaussian noise
seg::Image make_image(int width, int height, int seed) {
    std::mt19937 rng(static_cast<unsigned>(seed));
    std::uniform_int_distribution<int> color(0, 255);
    std::uniform_int_distribution<int> cluster(0, kTrueClusters - 1);
    std::normal_distribution<double>   noise(0.0, 15.0);

    std::vector<std::array<int, 3>> centers(kTrueClusters);
    for (auto& c : centers) c = {color(rng), color(rng), color(rng)};

    seg::Image img;
    img.width    = width;
    img.height   = height;
    img.channels = 3;
    img.data.resize(static_cast<size_t>(width) * height * 3);
    for (size_t p = 0; p < img.data.size(); p += 3) {
        const auto& c = centers[cluster(rng)];
        for (int ch = 0; ch < 3; ++ch) {
            double v = c[ch] + noise(rng);
            img.data[p + ch] = static_cast<unsigned char>(std::clamp(v, 0.0, 255.0));
        }
    }
    return img;
}

double run_once(const std::string& algo, const seg::Image& src, int k, int& iters) {
    seg::Image img = src; // the algorithms work in place

    auto t0 = std::chrono::steady_clock::now();
    if (algo == "kmeans") iters = seg::kmeans_segment(img, k, kSeed, kIters);
    else                  iters = seg::pfcm_segment(img, k, kSeed, kIters);
    auto t1 = std::chrono::steady_clock::now();

    return std::chrono::duration<double, std::milli>(t1 - t0).count();
}

Timing measure(const std::string& algo, const seg::Image& src, int k) {
    std::vector<double> times;
    int    iters = 0;
    double spent = 0.0;
    while (static_cast<int>(times.size()) < kMaxReps && (times.empty() || spent < kBudgetMs)) {
        double ms = run_once(algo, src, k, iters);
        times.push_back(ms);
        spent += ms;
    }

    std::sort(times.begin(), times.end());
    size_t n = times.size();
    return {iters, static_cast<int>(n),
            (n % 2) ? times[n / 2] : (times[n / 2 - 1] + times[n / 2]) / 2.0,
            times.front()};
}

void warm_up() {
    seg::Image src = make_image(512, 512, kSeed);
    auto start = std::chrono::steady_clock::now();
    int  iters = 0;
    while (std::chrono::steady_clock::now() - start < std::chrono::seconds(2)) {
        run_once("kmeans", src, 8, iters);
    }
}

} // anonymous namespace

int main(int argc, char** argv) {
    Options opt = parse_args(argc, argv);

    warm_up();
    std::printf("algo,mp,width,height,k,iters,reps,median_ms,min_ms,ns_per_px_iter\n");

    for (int mp : opt.mps) {
        int side = side_for_mp(mp);
        seg::Image src = make_image(side, side, kSeed);
        double pixels = static_cast<double>(side) * side;

        for (const auto& algo : opt.algos) {
            for (int k : opt.ks) {
                std::fprintf(stderr, "%s %d MP k=%d ...", algo.c_str(), mp, k);
                Timing t = measure(algo, src, k);
                std::fprintf(stderr, " %.1f ms\n", t.median_ms);

                std::printf("%s,%d,%d,%d,%d,%d,%d,%.1f,%.1f,%.2f\n",
                            algo.c_str(), mp, side, side, k, t.iters, t.reps,
                            t.median_ms, t.min_ms,
                            t.median_ms * 1e6 / (pixels * std::max(t.iters, 1)));
                std::fflush(stdout);
            }
        }
    }
    return 0;
}
