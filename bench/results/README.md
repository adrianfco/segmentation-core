`baseline.csv`: single-threaded, before any optimization. AMD Ryzen 5 7535HS, GCC 13.3, Release build, `./build/bench_segmentation > bench/results/baseline.csv`.

`scaling.csv`: same machine and grid after the OpenMP work, one pass per thread count (1, 2, 4, 8, 12 on 6 cores with SMT), produced by `./build/bench_segmentation --threads N` and concatenated. It carries the `threads` column that `baseline.csv` predates.
