# Benchmarks

Each entry is the median of 3 compiled calls after a warm-up call, with tangent vectors of canonical norm 1.0. Batched rows use `jax.vmap` and report the time of the batch divided by its size. `joint_basis` is timed once, and its row is the time of the whole call.

- CPU: AMD Ryzen 9 7950X 16-Core Processor, limited to 8 threads
- GPU: NVIDIA RTX A6000. Before the run, its utilization was 98 %, and 36989 MiB of its 49140 MiB were in use.
- Python 3.13.15, jax 0.12.0.1, stiefel 0.1.0, float64
- Source: the pre-review implementation. These timings have not been rerun against the current code.

| operation | batch | CPU, per item | GPU, per item |
|---|---:|---:|---:|
| exp St(257, 15) | 1 | 0.198 ms | GPU busy, not measured |
| log St(257, 15) | 1 | 0.984 ms | GPU busy, not measured |
| exp St(257, 15) | 1000 | 0.173 ms | GPU busy, not measured |
| log St(257, 15) | 1000 | 0.965 ms | GPU busy, not measured |
| exp St(86000, 16) | 1 | 25.779 ms | GPU busy, not measured |
| log St(86000, 16) | 1 | 29.308 ms | GPU busy, not measured |
| exp St(86000, 16) | 32 | 35.965 ms | GPU busy, not measured |
| log St(86000, 16) | 32 | 37.504 ms | GPU busy, not measured |
| joint_basis of 128 x St(86000, 16) | 1 | 21186.543 ms | GPU busy, not measured |
| exp St(2048, 16) after compression | 1000 | 1.903 ms | GPU busy, not measured |

## Intrinsic clustering (12 November 2026)

For the comparison, we generate eight balanced groups of frames in St(512, 2).
There are 128 observations in all, whose columns lie in a common subspace of dimension 12.
The data are generated once, with seed 2026, and the solvers are then run with seeds 0, 1 and 2
while keeping those observations fixed. The tolerance on the intrinsic gradient is `1e-6`
for both methods.

Lloyd is initialized by geodesic k-means++, and its mean solves omit zero memberships.
For e-means, five projected starts are compared before training for five epochs, with
`batch_size=32` and `decay=0.9`; five steps are allowed for the final refinement.
Every run satisfied its final convergence checks.

- CPU: AMD Ryzen 9 7950X 16-Core Processor; shared machine, CPU execution only.
- Python 3.13.15, JAX 0.12.0.1, float64.
- `OMP_NUM_THREADS=8` and `OPENBLAS_NUM_THREADS=8`; JAX uses its default CPU runtime.
- Timed jobs ran sequentially. Riemannian e-means was remeasured separately after the final PRNG correction.
- Each warm timing follows a first call. Timings include validation, compression, initialization and refinement.
- The common low-dimensional support and separated clusters favour compression and assignment pruning.

| metric and maximum tangent radius | Lloyd, median warm time | Riemannian e-means, median warm time | ratio | Lloyd intrinsic loss | e-means intrinsic loss |
|---|---:|---:|---:|---:|---:|
| canonical, 0.08 | 1.212682 s | 0.185947 s | 6.52 | 0.528207 | 0.528207 |
| Euclidean, 0.30 | 16.244683 s | 0.325293 s | 49.94 | 7.386476–17.943279 | 7.386476 |

The canonical losses differ by less than `2.5e-11`. In the wider Euclidean case, e-means matches Lloyd in two
seeds and has a lower loss in the third, where Lloyd reached another stationary partition. These are
training losses; no held-out or real-data comparison was made. The ratios describe the complete solvers,
including their different initialization rules, rather than an isolated effect of the EMA update.

The first observed canonical calls took 5.27 s for Lloyd and 5.38 s for e-means; the corresponding Euclidean
calls took 28.59 s and 15.54 s. First calls depend on prior compilation and are not independent cold-start
measurements. In particular, the warm speedup need not apply to a single clustering call.

### Per-seed warm measurements

| case | seed | Lloyd time (s) | e-means time (s) | Lloyd loss | e-means loss | e-means logarithms |
|---|---:|---:|---:|---:|---:|---:|
| canonical | 0 | 1.212682 | 0.192212 | 0.528207339385 | 0.528207339408 | 1152 |
| canonical | 1 | 1.217591 | 0.185947 | 0.528207339385 | 0.528207339409 | 1152 |
| canonical | 2 | 1.123017 | 0.150087 | 0.528207339385 | 0.528207339409 | 1152 |
| euclidean | 0 | 15.258265 | 0.325293 | 7.386475915777 | 7.386475915779 | 1280 |
| euclidean | 1 | 16.244683 | 0.260428 | 7.386475915778 | 7.386475915779 | 1280 |
| euclidean | 2 | 17.153854 | 0.352529 | 17.943278592190 | 7.386475915780 | 1280 |

Reproduce the datasets and comparisons with

```sh
OMP_NUM_THREADS=8 OPENBLAS_NUM_THREADS=8 uv run python benchmarks/clustering.py --seeds 3
OMP_NUM_THREADS=8 OPENBLAS_NUM_THREADS=8 uv run python benchmarks/clustering.py --seeds 3 --radius 0.3 --alpha=-0.5
```

The [algorithm note](../docs/fast-clustering.md) describes the geometry, literature and numerical restrictions.
