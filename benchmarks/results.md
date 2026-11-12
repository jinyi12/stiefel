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
