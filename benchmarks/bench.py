"""Wall-clock times of exp and log on CPU and GPU, and of the compressed path.

Run the script with ``uv run python benchmarks/bench.py``. It prints a table and writes it, with the hardware
and the versions, to ``benchmarks/results.md``. As the machine is shared, the script uses at most 8 CPU
threads, and it does not preallocate GPU memory.
"""

import os

# Thread and memory limits must be set before jax is imported.
os.environ["XLA_FLAGS"] = "--xla_cpu_multi_thread_eigen=true intra_op_parallelism_threads=8"
os.environ["OMP_NUM_THREADS"] = "8"
os.environ["XLA_PYTHON_CLIENT_PREALLOCATE"] = "false"

import platform  # noqa: E402
import statistics  # noqa: E402
import subprocess  # noqa: E402
import time  # noqa: E402
from pathlib import Path  # noqa: E402

import jax  # noqa: E402
import jax.numpy as jnp  # noqa: E402

import stiefel as sf  # noqa: E402

jax.config.update("jax_enable_x64", True)

# Each case is (n, p) with its vmap batch. With a batch of 32, every array of St(86000, 16) stays below 1 GB
# on a shared GPU.
CASES = [((257, 15), 1000), ((86000, 16), 32)]
COMPRESSED_FRAMES, COMPRESSED_EXPS = 128, 1000
DIST = 1.0
BUSY = "GPU busy, not measured"


def timed(fn, *args, repeats=3):
    """Median wall time of a compiled call, in seconds; a warm-up call compiles it first."""
    jax.block_until_ready(fn(*args))
    times = []
    for _ in range(repeats):
        start = time.perf_counter()
        jax.block_until_ready(fn(*args))
        times.append(time.perf_counter() - start)
    return statistics.median(times)


def inputs(key, n, p, batch):
    def one(k):
        k1, k2 = jax.random.split(k)
        base = sf.random_point(k1, n, p)
        t = sf.random_tangent(k2, base)
        return base, DIST * t / sf.norm(base, t)

    return jax.jit(jax.vmap(one))(jax.random.split(key, batch))


def bench_device(device):
    """Map each (operation, batch) to the time per item in seconds."""
    out = {}

    def record(op, batch, seconds):
        out[op, batch] = seconds
        print(device.platform, op, batch, f"{seconds * 1e3:.3f} ms", sep=" | ", flush=True)

    exp1 = jax.jit(sf.exp)
    log1 = jax.jit(lambda b, y: sf.log(b, y).tangent)
    expv = jax.jit(jax.vmap(sf.exp))
    logv = jax.jit(jax.vmap(lambda b, y: sf.log(b, y).tangent))
    with jax.default_device(device):
        for (n, p), batch in CASES:
            bases, tangents = inputs(jax.random.key(0), n, p, batch)
            points = expv(bases, tangents)
            record(f"exp St({n}, {p})", 1, timed(exp1, bases[0], tangents[0]))
            record(f"log St({n}, {p})", 1, timed(log1, bases[0], points[0]))
            record(f"exp St({n}, {p})", batch, timed(expv, bases, tangents) / batch)
            record(f"log St({n}, {p})", batch, timed(logv, bases, points) / batch)
            del bases, tangents, points

        # Compressed path: a joint basis of 128 frames of St(86000, 16), then 1000 exponentials of the small
        # frames. We time joint_basis once. It is not jit-compatible, because its output shape depends on the
        # data.
        frames, _ = inputs(jax.random.key(1), 86000, 16, COMPRESSED_FRAMES)
        start = time.perf_counter()
        jb = jax.block_until_ready(sf.joint_basis(frames))
        record(f"joint_basis of {COMPRESSED_FRAMES} x St(86000, 16)", 1, time.perf_counter() - start)
        del frames
        small = jb.coordinates[jnp.arange(COMPRESSED_EXPS) % COMPRESSED_FRAMES]
        keys = jax.random.split(jax.random.key(2), COMPRESSED_EXPS)
        tangents = jax.vmap(sf.random_tangent)(keys, small)
        k = jb.basis.shape[1]
        record(
            f"exp St({k}, 16) after compression",
            COMPRESSED_EXPS,
            timed(expv, small, tangents) / COMPRESSED_EXPS,
        )
    return out


def gpu_utilization():
    """Utilization in percent of the first GPU before the run and the nvidia-smi line, or None."""
    try:
        query = [
            "nvidia-smi",
            "--query-gpu=utilization.gpu,memory.used,memory.total",
            "--format=csv,noheader",
        ]
        line = subprocess.run(query, capture_output=True, text=True, check=True).stdout.splitlines()[0]
        return int(line.split(",")[0].strip().rstrip(" %")), line.strip()
    except (OSError, subprocess.CalledProcessError, IndexError, ValueError):
        return None, "unknown"


def environment(gpu_state):
    cpu = platform.processor()
    try:
        with open("/proc/cpuinfo") as f:
            cpu = next(line.split(":", 1)[1].strip() for line in f if line.startswith("model name"))
    except (OSError, StopIteration):
        pass
    lines = [f"- CPU: {cpu}, limited to 8 threads"]
    gpus = [d for d in jax.devices() if d.platform == "gpu"]
    if gpus:
        parts = [part.strip() for part in gpu_state.split(",")]
        if len(parts) == 3:
            utilization, used, total = parts
            state = (
                f"Before the run, its utilization was {utilization}, and {used} of its {total} were in use."
            )
        else:
            state = f"Its state before the run is {gpu_state}."
        lines.append(f"- GPU: {gpus[0].device_kind}. {state}")
    lines.append(
        f"- Python {platform.python_version()}, jax {jax.__version__}, stiefel {sf.__version__}, float64"
    )
    commit = subprocess.run(
        ["git", "rev-parse", "--short", "HEAD"], capture_output=True, text=True
    ).stdout.strip()
    lines.append(f"- git commit: {commit or 'none (uncommitted tree)'}")
    return "\n".join(lines)


def main():
    cpu = bench_device(jax.devices("cpu")[0])
    gpus = [d for d in jax.devices() if d.platform == "gpu"]
    utilization, gpu_state = gpu_utilization()
    gpu = {}
    if gpus:
        try:
            gpu = bench_device(gpus[0])
        except Exception as error:  # an out-of-memory error on a shared GPU is expected, not a bug
            print("GPU run failed:", type(error).__name__, str(error)[:200], flush=True)
        # If another job uses more than 50 % of the GPU, every GPU time is distorted, so we report no number.
        if utilization is None or utilization > 50:
            gpu = {}

    def cell(times, key):
        return f"{times[key] * 1e3:.3f} ms" if key in times else BUSY

    header = "| operation | batch | CPU, per item | GPU, per item |\n|---|---:|---:|---:|"
    body = "\n".join(f"| {op} | {b} | {cell(cpu, (op, b))} | {cell(gpu, (op, b))} |" for op, b in cpu)
    table = f"{header}\n{body}"
    print(table)
    text = (
        "# Benchmarks\n\n"
        f"Each entry is the median of 3 compiled calls after a warm-up call, with tangent vectors of "
        f"canonical norm {DIST}. Batched rows use `jax.vmap` and report the time of the batch divided by its "
        "size. "
        "`joint_basis` is timed once, and its row is the time of the whole call.\n\n"
        f"{environment(gpu_state)}\n\n{table}\n"
    )
    Path(__file__).with_name("results.md").write_text(text)


if __name__ == "__main__":
    main()
