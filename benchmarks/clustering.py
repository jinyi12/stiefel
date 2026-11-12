"""Compare intrinsic Lloyd clustering and the Riemannian EMA adaptation on the CPU.

Run ``uv run python benchmarks/clustering.py``. The printed JSON includes compilation, warm execution,
the intrinsic objective, convergence and EMA logarithm counts. No GPU is used and no files are written.
"""

import os

os.environ["CUDA_VISIBLE_DEVICES"] = ""
os.environ["JAX_PLATFORMS"] = "cpu"

import argparse  # noqa: E402
import json  # noqa: E402
import math  # noqa: E402
import platform  # noqa: E402
import time  # noqa: E402

import jax  # noqa: E402
import jax.numpy as jnp  # noqa: E402
import numpy as np  # noqa: E402

import stiefel as sf  # noqa: E402


def dataset(seed, n, p, rank, k, per, radius, alpha):
    rng = np.random.default_rng(seed)
    base = np.linalg.qr(rng.standard_normal((rank, p)))[0]
    groups = []
    for _ in range(k):
        centre = sf.polar(base + 0.5 * rng.standard_normal((rank, p)))
        noise = jnp.asarray(rng.standard_normal((per, rank, p)))
        noise = jax.vmap(lambda v, c=centre: sf.project_tangent(c, v))(noise)
        noise -= noise.mean(axis=0)
        lengths = jax.vmap(lambda v, c=centre: sf.norm(c, v, alpha=alpha))(noise)
        noise *= radius / lengths.max()
        groups.append(jax.vmap(lambda v, c=centre: sf.exp(c, v, alpha=alpha))(noise))
    small = jnp.concatenate(groups)
    basis = jnp.asarray(np.linalg.qr(rng.standard_normal((n, rank)))[0])
    return sf.lift(basis, small)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--n", type=int, default=512)
    parser.add_argument("--p", type=int, default=2)
    parser.add_argument("--rank", type=int, default=12)
    parser.add_argument("--clusters", type=int, default=8)
    parser.add_argument("--per-cluster", type=int, default=16)
    parser.add_argument("--radius", type=float, default=0.08)
    parser.add_argument("--alpha", type=float, default=0.0)
    parser.add_argument("--seeds", type=int, default=3)
    args = parser.parse_args()
    if not (args.n >= args.rank >= args.p >= 1 and args.rank * args.p > args.p * (args.p + 1) // 2):
        parser.error("require n >= rank >= p >= 1 and a positive-dimensional Stiefel manifold")
    if args.clusters < 1 or args.per_cluster < 2 or args.seeds < 1:
        parser.error("require clusters >= 1, per-cluster >= 2 and seeds >= 1")
    if (
        not math.isfinite(args.alpha)
        or args.alpha <= -1
        or not math.isfinite(args.radius)
        or args.radius <= 0
    ):
        parser.error("require finite alpha > -1 and radius > 0")
    jax.config.update("jax_enable_x64", True)
    points = dataset(
        2026, args.n, args.p, args.rank, args.clusters, args.per_cluster, args.radius, args.alpha
    )
    points.block_until_ready()
    assert bool(jnp.all(jax.vmap(sf.is_frame)(points)))
    report = {
        "configuration": vars(args),
        "python": platform.python_version(),
        "jax": jax.__version__,
        "runs": [],
    }
    for seed in range(args.seeds):
        for name, method in [("lloyd", sf.kmeans), ("emeans", sf.emeans)]:

            def run(fn=method, run_seed=seed):
                return jax.block_until_ready(
                    fn(jax.random.key(run_seed), points, args.clusters, alpha=args.alpha, tol=1e-6)
                )

            start = time.perf_counter()
            result = run()
            first = time.perf_counter() - start
            start = time.perf_counter()
            result = run()
            warm = time.perf_counter() - start
            own = jax.vmap(lambda c, x: sf.distance(c, x, alpha=args.alpha))(
                result.centers[result.labels], points
            )
            np.testing.assert_allclose(result.inertia, jnp.sum(own**2), rtol=1e-7, atol=1e-9)
            report["runs"].append(
                {
                    "seed": seed,
                    "method": name,
                    "first_seconds": first,
                    "warm_seconds": warm,
                    "inertia": float(result.inertia),
                    "converged": bool(result.converged),
                    "iterations": int(result.iterations),
                    "log_evaluations": int(result.log_evaluations) if name == "emeans" else None,
                }
            )
            print(json.dumps(report["runs"][-1]), flush=True)
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
