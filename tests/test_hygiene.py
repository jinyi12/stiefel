"""S4 and S8 in a fresh interpreter with JAX's defaults (float32), because conftest.py switches on float64."""

import json
import os
import subprocess
import sys

SCRIPT = r"""
import contextlib, io, json, warnings
import jax
before = dict(jax.config.values)
out, err = io.StringIO(), io.StringIO()
redirect = contextlib.redirect_stdout(out), contextlib.redirect_stderr(err)
with redirect[0], redirect[1], warnings.catch_warnings(record=True) as w:
    warnings.simplefilter("always")
    import jax.numpy as jnp
    import stiefel as sf
    k = jax.random.key(0)
    x, y = sf.random_point(k, 10, 3), sf.random_point(jax.random.key(1), 10, 3)
    t = sf.random_tangent(jax.random.key(2), x, alpha=1.0)
    near = sf.exp(x, 0.3 * t / sf.norm(x, t))
    pts = jnp.stack([x, near, sf.exp(x, -0.2 * t / sf.norm(x, t))])
    m = sf.Stiefel(10, 3, alpha=-0.5)
    results = {
        "random_point": x, "random_tangent": t, "exp": near, "log": sf.log(x, near).tangent,
        "distance": sf.distance(x, near), "inner": sf.inner(x, t, t), "norm": sf.norm(x, t),
        "project_tangent": sf.project_tangent(x, y), "polar": sf.polar(x + 0.1),
        "retract_qr": sf.retract_qr(x, t), "retract_polar": sf.retract_polar(x, t),
        "is_frame": sf.is_frame(x).astype(x.dtype), "joint_basis": sf.joint_basis(pts).basis,
        "lift": sf.lift(jnp.eye(10), x), "frechet_mean": sf.frechet_mean(pts).mean,
        "pairwise_distance": sf.pairwise_distance(pts), "kmeans": sf.kmeans(k, pts, 2).centers,
        "emeans": sf.emeans(k, pts, 2, epochs=1).centers,
        "Stiefel.random_point": m.random_point(k), "Stiefel.log": m.log(x, near),
        "Stiefel.dist": m.dist(x, near),
        "Stiefel.exp": m.exp(x, t), "Stiefel.retraction": m.retraction(x, t),
        "Stiefel.egrad2rgrad": m.euclidean_to_riemannian_gradient(x, y),
    }
    try:
        adapter = m.to_pymanopt(k)
        results["to_pymanopt"] = jnp.asarray(adapter.random_point())
        results["random_point after to_pymanopt"] = sf.random_point(k, 10, 3)
    except ImportError:
        pass
    after = dict(jax.config.values)
print(json.dumps({
    "changed": sorted(name for name in before if before[name] != after.get(name)),
    "dtypes": {name: str(v.dtype) for name, v in results.items()},
    "stdout": out.getvalue(), "stderr": err.getvalue(), "warnings": [str(x.message) for x in w],
}))
"""


def test_s4_and_s8_with_jax_defaults():
    # The test runs on the CPU, as the rest of the suite does (conftest.py), because the shared GPU ran out
    # of memory.
    env = dict(os.environ, JAX_PLATFORMS="cpu")
    run = subprocess.run([sys.executable, "-c", SCRIPT], capture_output=True, text=True, timeout=600, env=env)
    assert run.returncode == 0, run.stderr
    report = json.loads(run.stdout.strip().splitlines()[-1])
    # C1: the import of pymanopt switched jax_enable_x64 on. A public callable must not change the
    # configuration.
    assert report["changed"] == []
    assert set(report["dtypes"].values()) == {"float32"}, report["dtypes"]
    assert report["stdout"] == "" and report["stderr"] == "" and report["warnings"] == []
    assert run.stderr == ""
