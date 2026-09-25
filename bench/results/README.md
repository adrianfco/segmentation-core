`baseline.csv`: single-threaded, before any optimization. AMD Ryzen 5 7535HS, GCC 13.3, Release build, `./build/bench_segmentation > bench/results/baseline.csv`.

`scaling.csv`: same machine and grid after the OpenMP work, one pass per thread count (1, 2, 4, 8, 12 on 6 cores with SMT), produced by `./build/bench_segmentation --threads N` and concatenated. It carries the `threads` column that `baseline.csv` predates.

`counters.csv`: both algorithms with hardware counters, single-threaded, same machine. Needs `kernel.perf_event_paranoid` at 2 or lower. PFCM stops at 4 MP, 16 MP with k=16 runs ~12 min per rep.

```
./build/bench_segmentation --algo kmeans --mp 1,4,16 --k 4,8,16 --threads 1
./build/bench_segmentation --algo pfcm   --mp 1,4    --k 4,8,16 --threads 1
```

`kmeanspp.csv`: the same KMeans grid after the O(nk) seeding change, to compare against the KMeans rows of `counters.csv`.

`pfcm_pow.csv`: the same PFCM grid after the `std::pow` rewrite, to compare against the PFCM rows of `counters.csv`.
