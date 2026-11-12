"""Reference arrays from geomstats' canonical metric on St(n, p), for ``tests/test_geomstats_reference.py``.

geomstats 2.8 imports ``numpy.trapz``, which numpy 2 removed, so this script runs in its own environment:

    uv venv .venv-geomstats-ref --python 3.11
    uv pip install --python .venv-geomstats-ref/bin/python geomstats "numpy<2" scipy
    .venv-geomstats-ref/bin/python tools/geomstats_reference.py

It does not import ``stiefel``. Every logarithm of geomstats is checked independently, in two ways. First,
the exponential of Edelman, Arias and Smith (1998, Corollary 2.2) is evaluated with ``scipy.linalg.expm`` at
the returned tangent vector and compared with the endpoint. Second, the tangent vector is compared with the
one that produced the endpoint. ``{key}_log_ok`` records whether both checks pass to 1e-10.
"""

import sys
import warnings
from pathlib import Path

import geomstats
import numpy as np
import scipy
import scipy.linalg
from geomstats.geometry.stiefel import Stiefel

SHAPES = [(3, 2), (6, 3), (12, 4)]
DISTANCES = [0.3, 1.0]
LOG_TOL = 1e-12  # geomstats' default stopping tolerance on |C| is 1e-8; the comparison is to 1e-8.
OUT = Path(__file__).resolve().parents[1] / "tests" / "golden" / "geomstats.npz"


def canonical_inner(x, a, b):
    """tr(a^T (I - x x^T / 2) b), written out independently of geomstats."""
    return np.sum(a * b) - 0.5 * np.sum((x.T @ a) * (x.T @ b))


def random_tangent(rng, x):
    n, p = x.shape
    g = rng.standard_normal((p, p))
    z = rng.standard_normal((n, p))
    return x @ (g - g.T) / 2 + z - x @ (x.T @ z)


def eas_exp(x, d):
    """Edelman, Arias and Smith (1998), Corollary 2.2, with a full-column-rank QR of the normal part."""
    p = x.shape[1]
    a = x.T @ d
    q, r = np.linalg.qr(d - x @ a)
    k = q.shape[1]
    block = np.block([[a, -r.T], [r, np.zeros((k, k))]])
    mn = scipy.linalg.expm(block)[:, :p]
    return x @ mn[:p] + q @ mn[p:]


def main():
    rng = np.random.default_rng(20261111)
    out = {"distances": np.array(DISTANCES)}
    calls = [
        f"geomstats {geomstats.__version__}, numpy {np.__version__}, scipy {scipy.__version__}, "
        f"python {sys.version.split()[0]}, backend numpy",
        "space = geomstats.geometry.stiefel.Stiefel(n, p); metric = space.metric (StiefelCanonicalMetric)",
        "metric.inner_product(a, b, x): source formula tr(a^T (I - 0.5 x x^T) b), the canonical metric with "
        "no extra factor; {key}_inner_check is the same formula in numpy",
        "{key}_exp = metric.exp(tangent, base), tangent of canonical norm 1.0",
        "{key}_retraction = metric.retraction(tangent, base) (QR with the signs of diag(R) made positive)",
        f"{{key}}_log[i] = metric.log(log_point[i], base) with metric.log_solver.tol = {LOG_TOL} "
        "(default 1e-8) and max_iter = 500; log_point[i] = metric.exp(log_true[i], base), log_true[i] of "
        "canonical norm distances[i]",
        "{key}_log_residual[i] = |eas_exp(base, log[i]) - log_point[i]|_F with scipy.linalg.expm; "
        "{key}_log_error[i] = |log[i] - log_true[i]|_F; {key}_log_ok[i] = both <= 1e-10 and no warning",
        "base = Q factor of a standard normal (n, p) matrix with positive diag(R); tangents are "
        "x A + (I - x x^T) Z with A = skew(G), G and Z standard normal; numpy default_rng(20261111)",
    ]
    for n, p in SHAPES:
        key = f"n{n}_p{p}"
        space = Stiefel(n, p)
        metric = space.metric
        metric.log_solver.tol = LOG_TOL
        q, r = np.linalg.qr(rng.standard_normal((n, p)))
        x = q * np.where(np.diag(r) < 0, -1.0, 1.0)
        d = random_tangent(rng, x)
        d = d / np.sqrt(canonical_inner(x, d, d))
        a, b = random_tangent(rng, x), random_tangent(rng, x)
        out[f"{key}_base"] = x
        out[f"{key}_tangent"] = d
        out[f"{key}_exp"] = metric.exp(d, x)
        out[f"{key}_retraction"] = metric.retraction(d, x)
        out[f"{key}_inner_a"], out[f"{key}_inner_b"] = a, b
        out[f"{key}_inner"] = np.asarray(metric.inner_product(a, b, x))
        out[f"{key}_inner_check"] = np.asarray(canonical_inner(x, a, b))

        true, points, logs, residuals, errors, oks = [], [], [], [], [], []
        for dist in DISTANCES:
            t = random_tangent(rng, x)
            t = dist * t / np.sqrt(canonical_inner(x, t, t))
            y = metric.exp(t, x)
            with warnings.catch_warnings(record=True) as caught:
                warnings.simplefilter("always")
                try:
                    lg = metric.log(y, x)
                except ValueError as failure:  # geomstats raises when logm has a large imaginary part
                    print(f"{key} distance {dist}: geomstats raised ValueError: {failure}")
                    lg = np.full_like(t, np.nan)
            residual = np.linalg.norm(eas_exp(x, lg) - y) if np.all(np.isfinite(lg)) else np.inf
            error = np.linalg.norm(lg - t)
            true.append(t)
            points.append(y)
            logs.append(lg)
            residuals.append(residual)
            errors.append(error)
            oks.append(not caught and residual <= 1e-10 and error <= 1e-10)
            print(
                f"{key} distance {dist}: residual {residual:.2e}, error {error:.2e}, warnings {len(caught)}"
            )
        out[f"{key}_log_true"] = np.stack(true)
        out[f"{key}_log_point"] = np.stack(points)
        out[f"{key}_log"] = np.stack(logs)
        out[f"{key}_log_residual"] = np.array(residuals)
        out[f"{key}_log_error"] = np.array(errors)
        out[f"{key}_log_ok"] = np.array(oks)
    out["provenance"] = np.array(calls)
    np.savez(OUT, **out)
    print(f"wrote {OUT}")


if __name__ == "__main__":
    main()
