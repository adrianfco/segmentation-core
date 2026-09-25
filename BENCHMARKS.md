# Benchmarks

AMD Ryzen 5 7535HS (6 cores + SMT, 16 MiB L3), GCC 13.3, Release build.
`bench_segmentation` times the segmentation call only, median of up to 3 reps;
image synthesis and the per-rep copy are outside the timer. Single-threaded
runs also read hardware counters over that same region. CSVs in
[`bench/results/`](bench/results/).

## Thread scaling

16 MP, k=16, on 6 physical cores, measured at the OpenMP commit and not redone
since (`scaling.csv`):

| threads | KMeans ns/px/iter | speedup | PFCM ns/px/iter | speedup |
|--------:|------------------:|--------:|----------------:|--------:|
| 1       | 41.5              | 1.00    | 4298            | 1.00    |
| 2       | 23.4              | 1.78    | 2304            | 1.87    |
| 4       | 14.5              | 2.86    | 1413            | 3.04    |
| 8       | 12.7              | 3.28    | 1004            | 4.28    |
| 12      | 12.7              | 3.27    | 905             | 4.75    |

PFCM keeps gaining past 8 threads and KMeans does not. At 1 MP k=16 KMeans is
also slower on 8 threads (13.74 ns/px/iter) than on 4 (13.08), and faster again
on 12. Not chased down.

Output is bit-identical at every thread count: float sums are accumulated in
fixed 65536-pixel chunks and merged in chunk order, so the result does not
depend on how OpenMP splits the loop (`src/parallel.hpp`).

## KMeans++ seeding, O(nk²) to O(nk)

Seeding measured each pixel against every centroid chosen so far, once per new
centroid: `n·k(k-1)/2` distance evaluations. Keeping the closest distance
between rounds and measuring only against the centroid just added makes it
`n·(k-1)`.

Instructions retired per pixel per iteration, 16 MP, before (`counters.csv`)
and after (`kmeanspp.csv`):

| k  | before | after | ns/px/iter before | after | speedup |
|---:|-------:|------:|------------------:|------:|--------:|
| 4  | 137.4  | 125.5 | 10.18             | 9.21  | 1.11    |
| 8  | 272.6  | 212.5 | 18.87             | 13.87 | 1.36    |
| 16 | 654.0  | 387.1 | 45.35             | 25.64 | 1.77    |

The remaining count is linear in k at `38.2 + 21.8k`. Charging the old seeding
its `k(k-1)/2` evaluations per pixel, amortized over the benchmark's 10
iterations, gets back to the old numbers within 1%:

| k  | 38.2 + 21.8k + 21.8·k(k-1)/20 | measured |
|---:|------------------------------:|---------:|
| 4  | 138.5                         | 137.4    |
| 8  | 273.6                         | 272.6    |
| 16 | 648.2                         | 654.0    |

`l1d_mpki` goes from 1.21 to 2.05 at k=16: the same memory traffic over 41%
fewer instructions.

## PFCM and std::pow

`std::pow` was called `n·k(k+4)` times per iteration, `k²` of that in the
membership update. `m` and `eta` are runtime arguments, so none of it folded
away at compile time, and at k=16 it accounted for practically the whole
instruction count: 320 calls per pixel per iteration against 52376
instructions, about 164 each.

Two things changed. At the default `m = eta = 2` every exponent is 1 or 2,
which is a multiply. And `1 / sum_j (d_i/d_j)^e` factors into
`d_i^-e / sum_j d_j^-e`, so the inner sum comes out of the i loop and the
membership update drops from O(k²) to O(k).

1 MP, before (`counters.csv`) and after (`pfcm_pow.csv`):

| k  | instr/px/iter before | after | ns/px/iter before | after | speedup |
|---:|---------------------:|------:|------------------:|------:|--------:|
| 4  | 5526                 | 892   | 472.7             | 60.1  | 7.9     |
| 8  | 16058                | 1655  | 1386.0            | 116.9 | 11.9    |
| 16 | 52376                | 3183  | 4764.2            | 238.9 | 19.9    |

What is left is linear in k at `129 + 191k`. Repeat runs of the same build
drift by a few percent on timing; the instruction counts do not move.

`l1d_mpki` goes from 0.56 to 4.76 at k=16. The u and t arrays are still n·k
doubles each, so the traffic did not change while the instruction count fell
16x. IPC goes up rather than down, 2.90 to 3.69, so it is not stalling on that
traffic yet.

Output is unchanged on everything tested, at the default exponents and at
m and eta of 1.2 to 4.0, but the factored sum reassociates the arithmetic, so
bit-identical results are not guaranteed the way they are for the seeding
change.

## Counters

`bench_segmentation` opens cycles, instructions retired, branch misses and L1D
read misses through `perf_event_open`, enabled and disabled around the region
the stopwatch covers (`bench/counters.hpp`). They are read on the calling
thread, so only `--threads 1` runs carry them, and the columns stay empty
unless `kernel.perf_event_paranoid` is 2 or lower.

No memory-traffic column: the generic `LLC-*` events are unmapped on this CPU,
the per-core fill-source events credit prefetched lines to L2 and read near
zero on sequential access, and the L3 and data-fabric uncore PMUs are not
exposed on this part.

KMeans runs the same instruction stream per pixel at every image size with flat
`l1d_mpki`, but takes more cycles as the image grows:

| image | instr/px/iter | l1d_mpki | cycles/px/iter | IPC   |
|------:|--------------:|---------:|---------------:|------:|
| 1 MP  | 137.4         | 2.68     | 33.9           | 4.056 |
| 4 MP  | 137.4         | 2.68     | 34.1           | 4.030 |
| 16 MP | 137.4         | 2.68     | 35.8           | 3.839 |

1 MP of pixels is 3 MB and fits the 16 MiB L3, 16 MP is 48 MB and does not.
