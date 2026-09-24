`baseline.csv`: single-threaded, before any optimization. AMD Ryzen 5 7535HS, GCC 13.3, Release build, `./build/bench_segmentation > bench/results/baseline.csv`.

`scaling.csv`: same machine and grid after the OpenMP work, one pass per thread count (1, 2, 4, 8, 12 on 6 cores with SMT), produced by `./build/bench_segmentation --threads N` and concatenated. It carries the `threads` column that `baseline.csv` predates.

`counters.csv`: hardware counters for both algorithms, single-threaded, same machine. Needs `kernel.perf_event_paranoid` at 2 or lower, otherwise the counter columns come out empty. PFCM stops at 4 MP because 16 MP with k=16 takes about 12 minutes per repetition.

```
./build/bench_segmentation --algo kmeans --mp 1,4,16 --k 4,8,16 --threads 1
./build/bench_segmentation --algo pfcm   --mp 1,4    --k 4,8,16 --threads 1
```

There is no memory-traffic column. The generic `LLC-*` events are unmapped on this CPU, and the per-core fill-source events (`ls_any_fills_from_sys.*`) credit prefetched lines to L2, so they read near zero for sequential access; the L3 and data-fabric uncore PMUs that would answer the question are not exposed on this part at all. What the counters do show is that KMeans runs the same instruction stream per pixel at every image size while `cycles_per_px_iter` grows (33.9 to 35.8 at k=4, IPC 4.06 down to 3.84) with `l1d_mpki` flat, so the extra time at 16 MP is spent below L1.
