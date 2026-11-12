"""S9 at alpha = 0: agreement with geomstats' canonical metric, through stored arrays.

geomstats 2.8 cannot be imported under numpy 2. ``tests/golden/geomstats.npz`` was therefore computed in a
separate environment by ``tools/geomstats_reference.py``. Its ``provenance`` array lists the versions and the
calls.
geomstats' ``inner_product`` is ``tr(a^T (I - x x^T / 2) b)``, which is ``stiefel.inner`` at alpha = 0.
"""

from functools import cache
from pathlib import Path

import jax.numpy as jnp
import numpy as np
import pytest

import stiefel

GOLDEN = Path(__file__).parent / "golden" / "geomstats.npz"
SHAPES = [(3, 2), (6, 3), (12, 4)]
IDS = [f"n{n}_p{p}" for n, p in SHAPES]
ATOL = 1e-8


@cache
def golden():
    with np.load(GOLDEN) as data:
        return dict(data)


def arrays(n, p):
    g = golden()
    key = f"n{n}_p{p}"
    return {name[len(key) + 1 :]: value for name, value in g.items() if name.startswith(key + "_")}


@pytest.mark.parametrize(("n", "p"), SHAPES, ids=IDS)
def test_exp(n, p):
    a = arrays(n, p)
    out = stiefel.exp(jnp.asarray(a["base"]), jnp.asarray(a["tangent"]))
    np.testing.assert_allclose(out, a["exp"], rtol=0, atol=ATOL)


@pytest.mark.parametrize(("n", "p"), SHAPES, ids=IDS)
def test_inner(n, p):
    a = arrays(n, p)
    np.testing.assert_allclose(a["inner"], a["inner_check"], rtol=1e-12)
    val = stiefel.inner(jnp.asarray(a["base"]), jnp.asarray(a["inner_a"]), jnp.asarray(a["inner_b"]))
    np.testing.assert_allclose(val, a["inner"], rtol=0, atol=ATOL)


@pytest.mark.parametrize(("n", "p"), SHAPES, ids=IDS)
def test_retraction(n, p):
    a = arrays(n, p)
    out = stiefel.retract_qr(jnp.asarray(a["base"]), jnp.asarray(a["tangent"]))
    np.testing.assert_allclose(out, a["retraction"], rtol=0, atol=ATOL)


@pytest.mark.parametrize("i", range(2), ids=["dist0.3", "dist1.0"])
@pytest.mark.parametrize(("n", "p"), SHAPES, ids=IDS)
def test_log(n, p, i):
    a = arrays(n, p)
    res = stiefel.log(jnp.asarray(a["base"]), jnp.asarray(a["log_point"][i]))
    assert bool(res.converged)
    # Where geomstats' logarithm failed, the reference is the tangent vector that produced the endpoint. The
    # failure is at St(3, 2) and distance 0.3. There, the QR that geomstats takes of the rank-deficient
    # normal component gives a basis that is not orthogonal to the base, and geomstats raises ValueError.
    expected = a["log"][i] if a["log_ok"][i] else a["log_true"][i]
    np.testing.assert_allclose(res.tangent, expected, rtol=0, atol=ATOL)
    np.testing.assert_allclose(res.tangent, a["log_true"][i], rtol=0, atol=ATOL)


def test_reference_file_records_its_failures():
    g = golden()
    assert not g["n3_p2_log_ok"][0]
    assert all(g[f"n{n}_p{p}_log_ok"].all() for n, p in SHAPES[1:])
    assert any("geomstats 2.8" in line for line in g["provenance"])
