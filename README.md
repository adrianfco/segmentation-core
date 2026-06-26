# segmentation-core

A CPU image segmentation engine written in C++ with Python bindings via pybind11.

Supports **KMeans** and **PFCM** (Possibilistic Fuzzy C-Means) segmentation.  
Designed as a clean, testable library that a Python application can call directly.

---

## What it does

Given an image path, an algorithm, and a set of parameters, the engine:

1. Loads the input image from disk
2. Runs the selected segmentation algorithm
3. Writes the segmented image to a specified output path
4. Returns structured metadata (runtime, iterations, dimensions, success/error)

---

## Algorithms

| Algorithm | Key |
|-----------|-----|
| KMeans (KMeans++ init) | `"kmeans"` |
| Possibilistic Fuzzy C-Means | `"pfcm"` |

Both algorithms are seeded and reproducible.

---

## Installation

**Requirements**

- CMake ≥ 3.15
- A C++17 compiler (GCC, Clang, MSVC)
- Python ≥ 3.8 with development headers (`python3-dev`)

C++ dependencies (`stb_image`, `pybind11`) are fetched automatically at build time.

**Install directly from GitHub:**

```bash
pip install git+https://github.com/youruser/segmentation-core.git
```

**Or from a local clone:**

```bash
git clone https://github.com/youruser/segmentation-core.git
pip install ./segmentation-core
```

**For C++-only use** (no Python, e.g. embedding the library):

```bash
cmake -S . -B build -DCMAKE_BUILD_TYPE=Release
cmake --build build -j$(nproc)
```

---

## Python API

```python
from segmentation_core import segment_image

result = segment_image(
    image_path="input.png",
    output_path="output.png",
    algorithm="kmeans",  # or "pfcm"
    k=4,
    seed=42,
    max_iters=100,
)

print(result.success)      # bool
print(result.runtime_ms)   # float — wall time in milliseconds
print(result.iterations)   # int   — iterations executed
print(result.width)        # int
print(result.height)       # int
print(result.algorithm)    # str   — "kmeans" or "pfcm"
print(result.error_message)  # str — non-empty on failure
```

### PFCM-specific parameters

```python
result = segment_image(
    image_path="input.png",
    output_path="output.png",
    algorithm="pfcm",
    k=4,
    seed=42,
    max_iters=100,
    pfcm_m=2.0,    # fuzziness exponent (> 1)
    pfcm_eta=2.0,  # typicality exponent (> 1)
)
```

---

## C++ API

```cpp
#include "segmentation.hpp"

seg::SegmentationParams p;
p.input_path  = "input.png";
p.output_path = "output.png";
p.algorithm   = seg::Algorithm::KMeans;
p.k           = 4;
p.seed        = 42;
p.max_iters   = 100;

seg::SegmentationResult r = seg::segment_image(p);

if (!r.success) {
    std::cerr << r.error_message << "\n";
}
```

---

## Tests

**C++**

```bash
./build/test_segmentation
```

**Python**

```bash
python3 -m pytest tests/test_python_bindings.py -v
```

---

## Project layout

```
segmentation-core/
  CMakeLists.txt
  include/
    segmentation.hpp    — public API types and entry point
    image_io.hpp        — image load/save interface
    kmeans.hpp
    pfcm.hpp
  src/
    segmentation.cpp    — orchestration: validate → load → dispatch → save
    image_io.cpp        — stb_image-based I/O
    kmeans.cpp          — KMeans++ implementation
    pfcm.cpp            — PFCM implementation
    bindings.cpp        — pybind11 module
  tests/
    test_segmentation.cpp   — C++ tests
    test_python_bindings.py — Python tests
    fixtures/               — small PNG fixtures
  .github/workflows/
    ci.yml
```

---

## Adding a new algorithm

1. Add `include/myalgo.hpp` and `src/myalgo.cpp`
2. Add an entry to the `Algorithm` enum in `segmentation.hpp`
3. Add a dispatch branch in `segmentation.cpp`
4. Expose any new parameters on `SegmentationParams` if needed
5. Add tests
6. Update this README

---

## Roadmap

- Additional algorithms (SLIC superpixels, watershed, graph cut)
- Optional in-memory buffer API (skip disk I/O for embedded use)
- GPU/CUDA backend (separate module, same Python interface)
- Benchmark suite
