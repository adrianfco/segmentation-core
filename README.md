# segmentation-core

A C++17 library that segments images by color, with Python bindings. Each pixel
is assigned to one of `k` clusters and replaced by that cluster's color.
Currently, two clustering algorithms are available: KMeans and PFCM (Possibilistic Fuzzy
C-Means).

This library is the engine behind [segmentation-api](https://github.com/adrianfco/segmentation-api) distributed service.

## Performance

- **OpenMP parallelization** of every per-pixel loop: **4.75x on 6 cores**.
- **O(nk) KMeans++ seeding** (`src/centroids.hpp`): Fix naive kmeans++ implementation O(nk²) to 
  standard formulation O(nk). **KMeans is 1.77× faster** at k=16, with identical output.
- **Deterministic output**: floating-point sums are accumulated in fixed
  pixel chunks and merged in order, so the result is bit-identical for
  any thread count.
- **GIL released during the call** (`src/bindings.cpp`): `segment_image` drops the GIL and Python threads
  can segment images in parallel.

Measurements and hardware-counter results are in [BENCHMARKS.md](BENCHMARKS.md).

## Usage

```bash
pip install git+https://github.com/adrianfco/segmentation-core.git
```

```python
from segmentation_core import segment_image

r = segment_image(
    image_path="photo.jpg",
    output_path="segmented.png",  # .png, .jpg or .bmp
    algorithm="kmeans",           # or "pfcm"
    k=4,
    seed=42,
    max_iters=100,
    pfcm_m=2.0,                   # PFCM only, > 1
    pfcm_eta=2.0,                 # PFCM only, > 1
)

if r.success:
    print(f"{r.algorithm}: {r.width}x{r.height}, {r.iterations} iterations, {r.runtime_ms:.0f} ms")
else:
    print(r.error_message)
```

## Build and test

Needs a C++17 compiler, CMake 3.20+, Python 3.10+ with headers, git, and pytest.

```bash
cmake -B build          # set up the build/ folder
cmake --build build     # compile the targets
ctest --test-dir build  # run the C++ and Python tests
```

## Benchmarks

```bash
./build/bench_segmentation > results.csv             # 1, 4 and 16 MP images, k = 4, 8, 16
./build/bench_segmentation --algo pfcm --mp 1 --k 8   # one config, to the terminal
```

The full grid takes about a minute on 12 threads, and
needs ~4 GB of memory. Hardware counters need
`sudo sysctl kernel.perf_event_paranoid=2` on distributions that default to a
stricter level. Results and analysis are in [BENCHMARKS.md](BENCHMARKS.md).
