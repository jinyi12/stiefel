"""Regression tests against the legacy Stiefel routines (``Stiefel_Exp_Log.py``, ``Stiefel_Aux.py``).

``tests/golden/exp_log.npz`` was computed by ``tools/make_golden.py`` of the stochastic-rom repository from
the legacy reference snapshot dated 2026-11-02. Its ``provenance`` array names the legacy
function of every array. Tolerances are 1e-10 relative to the largest entry of the reference array, except
where the legacy logarithm itself is less accurate than that (see ``test_log``).
"""

from functools import cache
from pathlib import Path

import jax.numpy as jnp
import numpy as np
import pytest

import stiefel

GOLDEN = Path(__file__).parent / "golden" / "exp_log.npz"


@cache
def golden():
    with np.load(GOLDEN) as data:
        return dict(data)


SHAPES = [(12, 3), (50, 5), (5, 3)]
ALPHAS = [0.0, -0.5, 1.0]
NORMS = [0.1, 1.0, 1.5]
CASES = [(n, p, a, s) for n, p in SHAPES for a in range(3) for s in range(3)]
IDS = [f"n{n}_p{p}-alpha{ALPHAS[a]}-norm{NORMS[s]}" for n, p, a, s in CASES]


def case(n, p, a, s):
    g = golden()
    key = f"n{n}_p{p}"
    assert g["alphas"][a] == ALPHAS[a] and g["norms"][s] == NORMS[s]
    return g, key, float(g["alphas"][a])


def assert_rel(actual, expected, rtol=1e-10, atol=0.0):
    expected = np.asarray(expected, dtype=float)
    scale = max(np.abs(expected).max(initial=0.0), np.finfo(float).tiny)
    np.testing.assert_allclose(np.asarray(actual), expected, rtol=0, atol=max(rtol * scale, atol))


def legacy_tolerance(g, key, a, s):
    """The legacy logarithm meets its endpoint only to ``residual``; expect a tangent error of that size."""
    if not g[f"{key}_legacy_ok"][a, s]:
        pytest.skip("the legacy logarithm did not converge here (residual above 1e-8)")
    return 10.0 * float(g[f"{key}_residual"][a, s])


@pytest.mark.parametrize(("n", "p", "a", "s"), CASES, ids=IDS)
def test_exp(n, p, a, s):
    g, key, alpha = case(n, p, a, s)
    out = stiefel.exp(jnp.asarray(g[f"{key}_base"]), jnp.asarray(g[f"{key}_delta"][s]), alpha=alpha)
    assert_rel(out, g[f"{key}_exp"][a, s])


@pytest.mark.parametrize(("n", "p", "a", "s"), CASES, ids=IDS)
def test_log(n, p, a, s):
    g, key, alpha = case(n, p, a, s)
    atol = legacy_tolerance(g, key, a, s)
    res = stiefel.log(
        jnp.asarray(g[f"{key}_base"]), jnp.asarray(g[f"{key}_exp"][a, s]), alpha=alpha, tol=float(g["tau"])
    )
    assert bool(res.converged)
    assert_rel(res.tangent, g[f"{key}_log"][a, s], atol=atol)


@pytest.mark.parametrize(("n", "p"), SHAPES, ids=[f"n{n}_p{p}" for n, p in SHAPES])
@pytest.mark.parametrize("a", range(3), ids=[f"alpha{x}" for x in ALPHAS])
def test_inner(n, p, a):
    g, key, alpha = case(n, p, a, 0)
    val = stiefel.inner(
        jnp.asarray(g[f"{key}_base"]),
        jnp.asarray(g[f"{key}_inner_a"]),
        jnp.asarray(g[f"{key}_inner_b"]),
        alpha=alpha,
    )
    assert_rel(val, g[f"{key}_inner"][a])


@pytest.mark.parametrize(("n", "p", "a", "s"), CASES, ids=IDS)
def test_distance(n, p, a, s):
    g, key, alpha = case(n, p, a, s)
    atol = legacy_tolerance(g, key, a, s)
    d = stiefel.distance(
        jnp.asarray(g[f"{key}_base"]), jnp.asarray(g[f"{key}_exp"][a, s]), alpha=alpha, tol=float(g["tau"])
    )
    assert_rel(d, g[f"{key}_dist"][a, s], atol=atol)
