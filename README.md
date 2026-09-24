# segmentation-core

Color segmentation of images on the CPU, written in C++17 with Python bindings (pybind11).
Two algorithms: KMeans with KMeans++ init, and PFCM (Possibilistic Fuzzy C-Means).

`segment_image` loads an image, clusters its pixels by color, writes the result
(each pixel replaced by its cluster's color) and returns runtime, iteration count
and image size. Results are deterministic for a given seed.

This is the engine behind [segmentation-api](https://github.com/adrianfco/segmentation-api).

## Install

Prebuilt wheels for Linux x86_64 (CPython 3.10 to 3.13) are attached to each
[release](https://github.com/adrianfco/segmentation-core/releases):

```bash
pip install ./segmentation_core-<version>-<tags>.whl
```

From source (needs CMake >= 3.15, a C++17 compiler and Python headers;
stb and pybind11 are fetched by CMake):

```bash
pip install git+https://github.com/adrianfco/segmentation-core.git
```

## Python

```python
from segmentation_core import segment_image

r = segment_image(
    image_path="input.png",
    output_path="output.png",  # .png, .jpg or .bmp
    algorithm="kmeans",        # or "pfcm"
    k=4,
    seed=42,
    max_iters=100,
    pfcm_m=2.0,                # PFCM only: fuzziness exponent, > 1
    pfcm_eta=2.0,              # PFCM only: typicality exponent, > 1
)

if not r.success:
    print(r.error_message)
print(r.runtime_ms, r.iterations, r.width, r.height)
```

Bad inputs (missing file, `k < 1`, unwritable output) come back as
`success=False` with `error_message` set. An unknown `algorithm` raises `ValueError`.

`segment_image` releases the GIL while it runs.

## C++

```cpp
#include "segmentation.hpp"

seg::SegmentationParams p;
p.input_path  = "input.png";
p.output_path = "output.png";
p.algorithm   = seg::Algorithm::PFCM;
p.k           = 4;

seg::SegmentationResult r = seg::segment_image(p);
```

`kmeans_segment` and `pfcm_segment` (`include/kmeans.hpp`, `include/pfcm.hpp`)
work directly on an in-memory `seg::Image` if you don't want file I/O.

## Parallelism

The per-pixel loops of both algorithms are parallelized with OpenMP, and
`OMP_NUM_THREADS` sets the thread count. Partial sums are accumulated in
fixed-size pixel chunks and merged in chunk order, so the output is
bit-identical whatever the thread count, and identical again to a serial build. 
OpenMP is used when CMake finds it and skipped otherwise.

## Build and test

```bash
cmake -S . -B build -DCMAKE_BUILD_TYPE=Release
cmake --build build -j
./build/test_segmentation

pip install . pytest
python tests/make_fixture.py
pytest tests/test_python_bindings.py
```

## Benchmarks

`bench_segmentation` times both algorithms on synthetic in-memory images
(1, 4 and 16 MP; k = 4, 8, 16), without disk I/O, and reports the median time
and ns per pixel per iteration.

Single-threaded runs also read hardware counters over the same region as the
timer, and report cycles and instructions per pixel per iteration, IPC, and L1D
read misses and branch misses per 1000 instructions. Those columns need
`kernel.perf_event_paranoid` at 2 or lower and come out empty otherwise.

```bash
./build/bench_segmentation > results.csv             # full grid, ~23 min on one core
./build/bench_segmentation --algo pfcm --mp 1 --k 8  # one config, to the terminal
./build/bench_segmentation --threads 4               # pin the run to 4 threads
```

Rows go to stdout as CSV and progress to stderr, so redirecting the rows to a
file still shows progress on screen. The heaviest config (PFCM, 16 MP, k=16)
takes ~12 min and ~4 GB.
Baseline numbers are in [`bench/results/`](bench/results/).
