import jax
import jax.numpy as jnp
import numpy as np
import pytest
import scipy.linalg as sl
from hypothesis import example, given, settings
from hypothesis import strategies as st

import stiefel as sf
from stiefel.geometry import _logm_orthogonal

ALPHAS = [0.0, -0.5, 1.0]


def frame_error(x):
    x = np.asarray(x)
    return np.max(np.abs(x.T @ x - np.eye(x.shape[1])))


def s1_bound(p):
    return 100 * p * np.finfo(np.float64).eps


def pair(seed, n, p, dist, alpha=0.0):
    """A frame, a tangent of Riemannian norm ``dist`` at it, and the end point of their geodesic."""
    k1, k2 = jax.random.split(jax.random.key(seed))
    base = sf.random_point(k1, n, p)
    tangent = sf.random_tangent(k2, base, alpha=alpha)
    tangent = tangent * (dist / sf.norm(base, tangent, alpha=alpha))
    return base, tangent, sf.exp(base, tangent, alpha=alpha)


@st.composite
def shapes(draw):
    """Shapes (n, p) that include n = p, n < 2p and p = 1."""
    p = draw(st.integers(1, 6))
    n = draw(st.sampled_from([p, p + 1, 2 * p - 1, 2 * p, 2 * p + 3]))
    return max(n, p), p


def eas_exp(base, tangent):
    """Edelman, Arias and Smith (1998), Corollary 2.2, eqs. (2.45)-(2.47), evaluated with SciPy."""
    u, d = np.asarray(base), np.asarray(tangent)
    p = u.shape[1]
    a = u.T @ d
    q, r = np.linalg.qr(d - u @ a)
    e = sl.expm(np.block([[a, -r.T], [r, np.zeros((p, p))]]))[:, :p]
    return u @ e[:p] + q @ e[p:]


# ---------------------------------------------------------------- frames, tangents, inner products


def test_is_frame():
    x = sf.random_point(jax.random.key(0), 7, 3)
    assert sf.is_frame(x)
    assert not sf.is_frame(x * 1.01)
    assert sf.is_frame(x * 1.01, atol=0.1)


def test_project_tangent():
    key = jax.random.key(1)
    base = sf.random_point(key, 9, 3)
    v = jax.random.normal(jax.random.key(2), (9, 3))
    d = sf.project_tangent(base, v)
    s = base.T @ d
    np.testing.assert_allclose(s, -s.T, atol=1e-14)
    np.testing.assert_allclose(d, v - base @ ((base.T @ v + v.T @ base) / 2), atol=1e-14)
    np.testing.assert_allclose(sf.project_tangent(base, d), d, atol=1e-14)


@pytest.mark.parametrize("alpha", ALPHAS)
def test_inner_matches_dense_formula(alpha):
    base = sf.random_point(jax.random.key(3), 8, 3)
    a = sf.random_tangent(jax.random.key(4), base)
    b = sf.random_tangent(jax.random.key(5), base)
    c = (2 * alpha + 1) / (2 * (alpha + 1))
    dense = np.trace(np.asarray(a).T @ (np.eye(8) - c * np.asarray(base @ base.T)) @ np.asarray(b))
    np.testing.assert_allclose(sf.inner(base, a, b, alpha=alpha), dense, rtol=1e-13)
    np.testing.assert_allclose(
        sf.norm(base, a, alpha=alpha) ** 2, sf.inner(base, a, a, alpha=alpha), rtol=1e-13
    )
    assert sf.norm(base, jnp.zeros_like(a), alpha=alpha) == 0.0


@pytest.mark.parametrize("alpha", ALPHAS)
def test_random_tangent_is_standard_normal(alpha):
    """E g(D, D) equals the dimension of the manifold for a standard normal tangent."""
    n, p = 6, 3
    base = sf.random_point(jax.random.key(6), n, p)
    keys = jax.random.split(jax.random.key(7), 4000)
    d = jax.vmap(lambda k: sf.random_tangent(k, base, alpha=alpha))(keys)
    s = jnp.einsum("np,mnq->mpq", base, d)
    np.testing.assert_allclose(s, -jnp.swapaxes(s, 1, 2), atol=1e-13)
    sq = jax.vmap(lambda t: sf.inner(base, t, t, alpha=alpha))(d)
    dim = n * p - p * (p + 1) / 2
    assert abs(float(sq.mean()) - dim) < 4 * float(sq.std()) / np.sqrt(len(keys))
    # The skew part alone carries p (p - 1) / 2 of the dimensions, whatever the metric.
    a_part = jax.vmap(lambda t: sf.inner(base, base @ (base.T @ t), base @ (base.T @ t), alpha=alpha))(d)
    assert abs(float(a_part.mean()) - p * (p - 1) / 2) < 0.15


def test_random_point_reproducible_and_dtype():
    a = sf.random_point(jax.random.key(8), 5, 2)
    np.testing.assert_array_equal(a, sf.random_point(jax.random.key(8), 5, 2))
    assert not np.allclose(a, sf.random_point(jax.random.key(9), 5, 2))
    assert sf.random_point(jax.random.key(8), 5, 2, dtype=jnp.float32).dtype == jnp.float32


# ---------------------------------------------------------------- S1 for every frame-returning callable


# The edge shapes are given as explicit examples, so that they stay covered with fewer random draws.
@settings(max_examples=12, deadline=None)
@given(shapes(), st.integers(0, 2**31 - 1), st.sampled_from(ALPHAS))
@example((4, 4), 0, 0.0)
@example((5, 3), 1, 1.0)
@example((7, 1), 2, -0.5)
@example((1, 1), 3, 0.0)
def test_s1_frames_stay_frames(shape, seed, alpha):
    n, p = shape
    k1, k2 = jax.random.split(jax.random.key(seed))
    base = sf.random_point(k1, n, p)
    assert frame_error(base) <= s1_bound(p)
    tangent = sf.random_tangent(k2, base, alpha=alpha)
    for out in (
        sf.exp(base, tangent, alpha=alpha),
        sf.exp(base, 3.0 * tangent, alpha=alpha),
        sf.retract_qr(base, tangent),
        sf.retract_polar(base, tangent),
        sf.polar(base + tangent),
    ):
        assert out.shape == (n, p)
        assert frame_error(out) <= s1_bound(p)


def test_retract_qr_positive_diagonal():
    base = sf.random_point(jax.random.key(10), 6, 3)
    t = sf.random_tangent(jax.random.key(11), base)
    q = sf.retract_qr(base, t)
    r = q.T @ (base + t)
    np.testing.assert_allclose(q @ r, base + t, atol=1e-13)
    assert np.all(np.diag(r) > 0) and np.allclose(np.tril(r, -1), 0, atol=1e-13)


def test_polar_is_nearest_frame():
    m = jax.random.normal(jax.random.key(12), (7, 3))
    u, _, vt = np.linalg.svd(np.asarray(m), full_matrices=False)
    np.testing.assert_allclose(sf.polar(m), u @ vt, atol=1e-13)
    base = sf.random_point(jax.random.key(13), 7, 3)
    t = sf.random_tangent(jax.random.key(14), base)
    np.testing.assert_allclose(sf.retract_polar(base, t), sf.polar(base + t), atol=1e-14)


# ---------------------------------------------------------------- exponential


@pytest.mark.parametrize("alpha", ALPHAS)
@pytest.mark.parametrize("shape", [(9, 3), (5, 4), (4, 4), (6, 1)])
def test_exp_zero_tangent(alpha, shape):
    base = sf.random_point(jax.random.key(15), *shape)
    np.testing.assert_allclose(sf.exp(base, jnp.zeros_like(base), alpha=alpha), base, atol=1e-15)


@pytest.mark.parametrize("seed", range(4))
def test_exp_matches_edelman_arias_smith(seed):
    base, tangent, _ = pair(seed, 12, 4, 1.3)
    np.testing.assert_allclose(sf.exp(base, tangent), eas_exp(base, tangent), atol=1e-13)


@pytest.mark.parametrize("alpha", ALPHAS)
def test_exp_matches_dense_formula_of_huper_markina_silva_leite(alpha):
    """Zimmermann and Hüper (2022), eq. (8): an n x n formula that the O(n p^2) evaluation must reproduce."""
    base, tangent, _ = pair(16, 7, 3, 1.1, alpha)
    u, d = np.asarray(base), np.asarray(tangent)
    a = u.T @ d
    big = -(2 * alpha + 1) / (alpha + 1) * u @ a @ u.T + d @ u.T - u @ d.T
    dense = sl.expm(big) @ u @ sl.expm(alpha / (alpha + 1) * a)
    np.testing.assert_allclose(sf.exp(base, tangent, alpha=alpha), dense, atol=1e-13)


@pytest.mark.parametrize("alpha", ALPHAS)
def test_exp_is_a_geodesic(alpha):
    """exp(base, (s + t) D) = exp(exp(base, s D), t D(s)) with D(s) the velocity at s."""
    base, tangent, _ = pair(17, 8, 3, 1.0, alpha)
    h = 1e-5
    mid = sf.exp(base, 0.5 * tangent, alpha=alpha)
    vel = (
        sf.exp(base, (0.5 + h) * tangent, alpha=alpha) - sf.exp(base, (0.5 - h) * tangent, alpha=alpha)
    ) / (2 * h)
    vel = sf.project_tangent(mid, vel)
    np.testing.assert_allclose(
        sf.exp(mid, 0.5 * vel, alpha=alpha), sf.exp(base, tangent, alpha=alpha), atol=1e-8
    )
    # Constant speed along the geodesic.
    np.testing.assert_allclose(sf.norm(mid, vel, alpha=alpha), sf.norm(base, tangent, alpha=alpha), rtol=1e-8)


@pytest.mark.parametrize("alpha", ALPHAS)
def test_exp_rank_deficient_normal_component(alpha):
    """A tangent whose normal component has rank 1 (and 0) for p = 3."""
    base = sf.random_point(jax.random.key(18), 10, 3)
    perp = sf.project_tangent(base, jax.random.normal(jax.random.key(19), (10, 3)))
    perp = perp - base @ (base.T @ perp)
    rank1 = jnp.outer(perp[:, 0], jnp.array([1.0, -2.0, 0.5]))
    g = jax.random.normal(jax.random.key(20), (3, 3))
    for tangent in (base @ (g - g.T) + rank1, base @ (g - g.T)):
        out = sf.exp(base, tangent, alpha=alpha)
        assert frame_error(out) <= s1_bound(3)
        u, d = np.asarray(base), np.asarray(tangent)
        a = u.T @ d
        big = -(2 * alpha + 1) / (alpha + 1) * u @ a @ u.T + d @ u.T - u @ d.T
        np.testing.assert_allclose(out, sl.expm(big) @ u @ sl.expm(alpha / (alpha + 1) * a), atol=1e-13)


# ---------------------------------------------------------------- matrix logarithm of orthogonal matrices


def _rotation(rng, angles, n):
    blocks = [np.array([[np.cos(t), -np.sin(t)], [np.sin(t), np.cos(t)]]) for t in angles]
    q = np.linalg.qr(rng.normal(size=(n, n)))[0]
    return q @ sl.block_diag(*blocks, np.eye(n - 2 * len(angles))) @ q.T


@pytest.mark.parametrize(
    "angles",
    [[], [0.7, 0.7, 0.7, -0.3], [1.0, 2.0, 3.0], [np.pi - 1e-2, 1.0], [np.pi - 1e-4, 0.5]],
    ids=["identity", "repeated", "spread", "near-pi-1e-2", "near-pi-1e-4"],
)
def test_logm_orthogonal_against_scipy(angles):
    rng = np.random.default_rng(0)
    v = _rotation(rng, angles, 10)
    log_v, ok = jax.jit(_logm_orthogonal)(jnp.asarray(v))
    assert ok
    tol = 1e-12 if not angles or max(angles) < 3.1 else 1e-11
    np.testing.assert_allclose(log_v, np.real(sl.logm(v)), atol=tol)


def test_logm_orthogonal_haar():
    rng = np.random.default_rng(1)
    for _ in range(5):
        g = rng.normal(size=(12, 12))
        v = sl.expm(g - g.T)
        log_v, ok = _logm_orthogonal(jnp.asarray(v))
        assert ok
        np.testing.assert_allclose(log_v, np.real(sl.logm(v)), atol=1e-12)


@pytest.mark.parametrize("gap", [0.0, 1e-9])
def test_logm_orthogonal_reports_eigenvalue_at_minus_one(gap):
    v = _rotation(np.random.default_rng(2), [np.pi - gap, 1.0], 8)
    log_v, ok = _logm_orthogonal(jnp.asarray(v))
    assert not ok
    reflection = np.diag([-1.0, 1, 1, 1])
    assert not _logm_orthogonal(jnp.asarray(reflection))[1]


# ---------------------------------------------------------------- logarithm


@pytest.mark.parametrize("alpha", ALPHAS)
@pytest.mark.parametrize("dist", [0.1, 1.0, 1.5])
@pytest.mark.parametrize("shape", [(12, 4), (5, 3), (6, 1)])
def test_exp_log_round_trip(alpha, dist, shape):
    for seed in range(3):
        base, tangent, point = pair(100 + seed, *shape, dist, alpha)
        res = sf.log(base, point, alpha=alpha)
        assert res.converged, (seed, res)
        np.testing.assert_allclose(res.tangent, tangent, atol=1e-10)
        assert res.residual <= 100 * 1e-12 * np.sqrt(shape[1])
        # Tangency to 1e3 eps relative accuracy.
        s = np.asarray(base.T @ res.tangent)
        assert np.max(np.abs(s + s.T)) <= 1e3 * np.finfo(float).eps * max(
            1.0, float(jnp.abs(res.tangent).max())
        )
        np.testing.assert_allclose(sf.exp(base, res.tangent, alpha=alpha), point, atol=1e-11)


@pytest.mark.parametrize("alpha", ALPHAS)
@pytest.mark.parametrize("shape", [(9, 3), (5, 4), (3, 3), (4, 1)])
def test_log_of_base_is_zero(alpha, shape):
    base = sf.random_point(jax.random.key(21), *shape)
    res = sf.log(base, base, alpha=alpha)
    assert res.converged
    np.testing.assert_allclose(res.tangent, 0.0, atol=1e-14)
    assert sf.distance(base, base, alpha=alpha) <= 1e-14


@pytest.mark.parametrize("alpha", ALPHAS)
def test_log_shared_columns(alpha):
    """Both frames contain the same first column, so the normal component of the point has rank p - 1."""
    base = sf.random_point(jax.random.key(22), 10, 3)
    perp = jax.random.normal(jax.random.key(23), (10, 2)) * 0.4
    perp = perp - base @ (base.T @ perp)
    tangent = jnp.concatenate([jnp.zeros((10, 1)), perp], axis=1)
    g = jnp.array([[0.0, 0.0, 0.0], [0.0, 0.0, -0.3], [0.0, 0.3, 0.0]])
    tangent = tangent + base @ g
    point = sf.exp(base, tangent, alpha=alpha)
    np.testing.assert_allclose(point[:, 0], base[:, 0], atol=1e-14)
    res = sf.log(base, point, alpha=alpha)
    assert res.converged
    np.testing.assert_allclose(res.tangent, tangent, atol=1e-10)


@pytest.mark.parametrize("alpha", ALPHAS)
def test_log_identical_column_spans(alpha):
    """The point is a rotation of the base inside its own span: the normal component is zero."""
    base = sf.random_point(jax.random.key(24), 8, 3)
    g = jnp.array([[0.0, -0.4, 0.1], [0.4, 0.0, -0.2], [-0.1, 0.2, 0.0]])
    point = sf.exp(base, base @ g, alpha=alpha)
    res = sf.log(base, point, alpha=alpha)
    assert res.converged
    np.testing.assert_allclose(res.tangent, base @ g, atol=1e-10)


@pytest.mark.parametrize("alpha", ALPHAS)
@pytest.mark.parametrize("shape", [(6, 1), (8, 2), (8, 3)])
def test_log_antipodal_columns_does_not_converge(alpha, shape):
    base = sf.random_point(jax.random.key(25), *shape)
    point = base.at[:, 0].multiply(-1.0)
    res = sf.log(base, point, alpha=alpha, max_iter=30)
    assert not res.converged
    assert np.all(np.isfinite(res.tangent))
    assert np.isnan(sf.distance(base, point, alpha=alpha, max_iter=30))


def test_log_non_convergence_within_max_iter():
    base, _, point = pair(26, 10, 3, 1.5)
    res = sf.log(base, point, max_iter=1)
    assert not res.converged and int(res.iterations) == 1
    assert np.all(np.isfinite(res.tangent))


def test_log_tolerance_validation():
    base, _, point = pair(27, 6, 2, 0.5)
    with pytest.raises(ValueError):
        sf.log(base, point, tol=1e-15)
    with pytest.raises(ValueError):
        sf.distance(base, point, tol=1e-15)
    assert sf.log(base, point, tol=1e-8).converged


@pytest.mark.parametrize("alpha", ALPHAS)
def test_distance_symmetric(alpha):
    for seed in range(3):
        a, _, b = pair(200 + seed, 9, 3, 1.2, alpha)
        dab, dba = sf.distance(a, b, alpha=alpha), sf.distance(b, a, alpha=alpha)
        np.testing.assert_allclose(dab, 1.2, rtol=1e-10)
        assert abs(dab - dba) <= 100 * 1e-12


def test_log_float32():
    base, tangent, point = pair(28, 10, 3, 0.5)
    res = sf.log(base.astype(jnp.float32), point.astype(jnp.float32))
    assert res.tangent.dtype == jnp.float32
    assert res.converged
    np.testing.assert_allclose(res.tangent, tangent, atol=1e-4)


# ---------------------------------------------------------------- transformations and devices (S5)


@pytest.mark.parametrize("alpha", ALPHAS)
def test_jit_and_vmap(alpha):
    keys = jax.random.split(jax.random.key(29), 6)
    bases = jax.vmap(lambda k: sf.random_point(k, 9, 3))(keys)
    tangents = jax.vmap(lambda k, b: sf.random_tangent(k, b, alpha=alpha))(keys, bases)
    tangents = (
        tangents * 0.8 / jax.vmap(lambda b, t: sf.norm(b, t, alpha=alpha))(bases, tangents)[:, None, None]
    )
    exp_v = jax.jit(jax.vmap(lambda b, t: sf.exp(b, t, alpha=alpha)))
    log_v = jax.jit(jax.vmap(lambda b, y: sf.log(b, y, alpha=alpha)))
    points = exp_v(bases, tangents)
    res = log_v(bases, points)
    assert res.converged.shape == (6,) and bool(res.converged.all())
    np.testing.assert_allclose(res.tangent, tangents, atol=1e-10)
    for i in range(6):
        np.testing.assert_allclose(points[i], sf.exp(bases[i], tangents[i], alpha=alpha), atol=1e-14)
    dist = jax.jit(jax.vmap(lambda b, y: sf.distance(b, y, alpha=alpha)))(bases, points)
    np.testing.assert_allclose(dist, 0.8, rtol=1e-10)


@pytest.mark.skipif(not any(d.platform == "gpu" for d in jax.devices()), reason="no CUDA device")
@pytest.mark.parametrize("alpha", [0.0, 1.0])
def test_exp_log_on_gpu(alpha):
    gpu = jax.devices("gpu")[0]
    base, tangent, point = pair(30, 20, 4, 1.0, alpha)
    base, tangent = jax.device_put(base, gpu), jax.device_put(tangent, gpu)
    point = jax.jit(lambda b, t: sf.exp(b, t, alpha=alpha))(base, tangent)
    res = jax.jit(lambda b, y: sf.log(b, y, alpha=alpha))(base, point)
    assert res.tangent.devices() == {gpu}
    assert res.converged
    np.testing.assert_allclose(np.asarray(res.tangent), np.asarray(tangent), atol=1e-10)
    bad = jax.jit(lambda b: sf.log(b, b.at[:, 0].multiply(-1.0), alpha=alpha, max_iter=20))(base)
    assert not bad.converged and np.all(np.isfinite(np.asarray(bad.tangent)))


def test_annotations_are_checked_at_run_time():
    """The import hook of conftest.py checks the jaxtyping shapes: base and tangent must share (n, p)."""
    base = sf.random_point(jax.random.key(31), 5, 2)
    with pytest.raises(Exception, match="(?i)type-check|shape|dimension"):
        sf.exp(base, jnp.zeros((5, 3)))


# ---------------------------------------------------------------- review findings: derivatives (W2, S5)


@pytest.mark.parametrize("alpha", ALPHAS)
def test_exp_derivatives_match_central_differences(alpha):
    """Forward and reverse mode through exp at three tangents.

    The tangents are the zero tangent, a generic tangent and a tangent whose normal component has rank 1.
    The QR of the normal component used to give NaN at the first and the last of these tangents.
    """
    base = sf.random_point(jax.random.key(40), 9, 3)
    perp = jax.random.normal(jax.random.key(41), (9, 3))
    perp = perp - base @ (base.T @ perp)
    g = jax.random.normal(jax.random.key(42), (3, 3))
    rank1 = base @ (g - g.T) / 2 + jnp.outer(perp[:, 0], jnp.array([1.0, -2.0, 0.5]))
    generic = 0.7 * sf.random_tangent(jax.random.key(43), base, alpha=alpha)
    direction = sf.random_tangent(jax.random.key(44), base, alpha=alpha)
    weights = jax.random.normal(jax.random.key(45), (9, 3))
    for start in (jnp.zeros_like(base), generic, rank1):

        def curve(s, start=start):
            return sf.exp(base, start + s * direction, alpha=alpha)

        forward = jax.jvp(curve, (0.0,), (1.0,))[1]
        h = 1e-5
        np.testing.assert_allclose(forward, (curve(h) - curve(-h)) / (2 * h), atol=1e-8)
        reverse = jax.grad(lambda d: jnp.sum(sf.exp(base, d, alpha=alpha) * weights))(start)
        assert np.all(np.isfinite(reverse))
        # Reverse mode is the transpose of forward mode.
        np.testing.assert_allclose(jnp.sum(reverse * direction), jnp.sum(forward * weights), atol=1e-12)
    # The value keeps the QR form.
    np.testing.assert_array_equal(
        sf.exp(base, generic, alpha=alpha), sf.geometry._exp_qr(base, generic, alpha)
    )


def test_log_and_distance_refuse_differentiation():
    base, _, point = pair(46, 6, 2, 0.5)
    for fn in (lambda y: jnp.sum(sf.log(base, y).tangent), lambda y: sf.distance(base, y)):
        with pytest.raises(TypeError, match="not differentiable"):
            jax.grad(fn)(point)
        with pytest.raises(TypeError, match="not differentiable"):
            jax.jvp(fn, (point,), (point,))
    with pytest.raises(TypeError, match="not differentiable"):
        jax.jacfwd(lambda b: sf.distance(b, point))(base)
    # Without differentiation the guard is the identity, also under jit and vmap.
    assert sf.log(base, point).converged


def test_exp_projects_a_non_tangent_input():
    base = sf.random_point(jax.random.key(47), 7, 3)
    v = jax.random.normal(jax.random.key(48), (7, 3))
    np.testing.assert_allclose(sf.exp(base, v), sf.exp(base, sf.project_tangent(base, v)), atol=1e-14)


# ---------------------------------------------------------------- review findings: S1 for long tangents (C4)


@pytest.mark.parametrize("alpha", ALPHAS)
def test_s1_at_the_stated_tangent_norm_limit(alpha):
    """exp keeps S1 up to the norm 500 that its docstring states (0.73 of the bound at worst, measured)."""
    for shape in [(8, 1), (10, 3), (6, 6), (12, 4)]:
        for seed in range(2):
            base, _, point = pair(50 + seed, *shape, 500.0, alpha)
            assert frame_error(point) <= s1_bound(shape[1]), (shape, seed)


# ---------------------------------------------------------------- review findings: log status and domain


def test_log_rejects_non_finite_tolerances_and_never_loosens_the_endpoint():
    """F1: tol = inf, NaN, 0 or -1 raises; tol = 1 or 1e-2 stops early but cannot report convergence."""
    base, _, point = pair(27, 6, 2, 0.5)
    for tol in (float("inf"), float("nan"), 0.0, -1.0):
        for fn in (sf.log, sf.distance):
            with pytest.raises(ValueError):
                fn(base, point, tol=tol)
    for tol in (1.0, 1e-2):
        res = sf.log(base, point, tol=tol)
        assert not res.converged and res.residual > 100 * np.sqrt(np.finfo(float).eps) * np.sqrt(2)
    res = sf.log(base, point, tol=1e-6)
    assert res.converged and res.residual <= 100 * 1e-6 * np.sqrt(2)


def test_log_status_validates_the_result_at_max_iter():
    """U3: St(5, 3) at distance 2.8 meets the endpoint bound when it stops at max_iter (residual 9.5e-12)."""
    base, tangent, point = pair(1000, 5, 3, 2.8)
    res = sf.log(base, point)
    assert int(res.iterations) == 100
    assert res.converged and res.residual <= 100 * 1e-12 * np.sqrt(3)
    np.testing.assert_allclose(eas_exp(base, res.tangent), point, atol=1e-10)


def test_log_iterations_count_both_variants():
    """U4: for alpha != 0 the fallback variant adds its iterations, at most 2 * max_iter in total."""
    base, _, point = pair(26, 10, 3, 1.5, 1.0)
    for max_iter in (1, 3):
        assert int(sf.log(base, point, alpha=1.0, max_iter=max_iter).iterations) <= 2 * max_iter
    assert int(sf.log(base, point, alpha=0.0, max_iter=1).iterations) <= 1


def defective(frame, defect, seed):
    """``frame @ (I + S)`` with a symmetric ``S``, so that ``max|x^T x - I|`` is about ``defect``."""
    p = frame.shape[1]
    g = jax.random.normal(jax.random.key(seed), (p, p))
    s = (g + g.T) / 2
    x = frame @ (jnp.eye(p) + defect * s / (2 * jnp.max(jnp.abs(s))))
    return x, float(jnp.max(jnp.abs(x.T @ x - jnp.eye(p))))


def test_log_frame_defect_cutoff():
    """U2: frames are accepted up to the defect max(min(tol, sqrt(eps)), 100 p eps), deterministically."""
    base, tangent, point = pair(60, 10, 3, 0.7)
    for target, accepted in [(0.9e-12, True), (1.2e-12, False), (3.4e-10, False)]:
        b, d = defective(base, target, 61)
        assert (d <= 1e-12) == accepted
        for res in (sf.log(b, b), sf.log(b, point)):
            assert bool(res.converged) == accepted, (target, res)
            assert np.all(np.isfinite(res.tangent))
        # With the re-orthonormalized completion, the iteration continues, and the tangent is right up to the
        # defect.
        np.testing.assert_allclose(sf.log(b, point).tangent, tangent, atol=1e3 * d)
    # A looser tolerance accepts a larger defect.
    b, _ = defective(base, 3.4e-10, 61)
    assert sf.log(b, point, tol=1e-9).converged


def _plane_rotation(theta, p=3):
    r = np.eye(p)
    r[:2, :2] = [[np.cos(theta), -np.sin(theta)], [np.sin(theta), np.cos(theta)]]
    return r


@pytest.mark.parametrize("shape", [(10, 3), (3, 3)])
def test_log_in_span_rotation_beyond_the_iteration_domain(shape):
    """C3: for alpha = -1/2, the iteration needs rotation angles below pi (1 + alpha) = pi / 2.

    A point in the span of the base is solved in closed form up to pi, including n = p.
    """
    base = sf.random_point(jax.random.key(62), *shape)
    for theta in (1.5, 1.6, 2.5, 3.1):
        r = _plane_rotation(theta)
        res = sf.log(base, base @ jnp.asarray(r), alpha=-0.5)
        assert res.converged, theta
        np.testing.assert_allclose(res.tangent, base @ np.real(sl.logm(r)), atol=1e-12)


def test_log_near_span_domain_boundary_alpha_minus_half():
    """C3: with a normal component, rotation angles just below pi / 2 converge, and angles just above do not.

    This is the documented domain theta < pi (1 + alpha). There is no false success on either side.
    """
    base = sf.random_point(jax.random.key(63), 10, 3)
    perp = jax.random.normal(jax.random.key(64), (10, 3))
    perp = perp - base @ (base.T @ perp)
    for theta, expected in [(1.55, True), (1.6, False)]:
        tangent = base @ np.real(sl.logm(_plane_rotation(theta))) + 1e-2 * perp
        point = sf.exp(base, tangent, alpha=-0.5)
        res = sf.log(base, point, alpha=-0.5)
        assert bool(res.converged) == expected, theta
        assert np.all(np.isfinite(res.tangent))
        if expected:
            np.testing.assert_allclose(res.tangent, tangent, atol=1e-10)


@pytest.mark.parametrize("alpha", ALPHAS)
def test_log_in_span_eigenvalue_minus_one(alpha):
    """A point base @ R in the span has no principal logarithm if det R = -1 or R has two eigenvalues -1.

    The first case is a reflection, and the second case is a rotation by pi. The result is finite and
    reported as not converged.
    """
    rng = np.random.default_rng(3)
    q = np.linalg.qr(rng.normal(size=(3, 3)))[0]
    v = np.array([1.0, 2.0, 2.0]) / 3
    for r in (np.eye(3) - 2 * np.outer(v, v), q @ np.diag([-1.0, -1.0, 1.0]) @ q.T):
        for n in (3, 10):
            base = sf.random_point(jax.random.key(65), n, 3)
            res = sf.log(base, base @ jnp.asarray(r), alpha=alpha, max_iter=30)
            assert not res.converged and np.all(np.isfinite(res.tangent))


# ---------------------------------------------------------------- review findings: distance


def test_distance_symmetry_over_a_range_of_distances():
    for dist in (0.05, 0.5, 1.0, 1.5, 2.0, 2.5):
        a, _, b = pair(70, 9, 3, dist)
        dab, dba = sf.distance(a, b), sf.distance(b, a)
        np.testing.assert_allclose(dab, dist, rtol=1e-10)
        assert abs(dab - dba) <= 100 * 1e-12, dist


@pytest.mark.parametrize("alpha", [0.0, -0.5])
def test_distance_triangle_inequality(alpha):
    centre = sf.random_point(jax.random.key(71), 8, 3)
    keys = jax.random.split(jax.random.key(72), 15)
    tangents = jnp.stack([sf.random_tangent(k, centre, alpha=alpha) for k in keys])
    tangents = tangents * (0.5 / jax.vmap(lambda t: sf.norm(centre, t, alpha=alpha))(tangents))[:, None, None]
    pts = jax.vmap(lambda t: sf.exp(centre, t, alpha=alpha))(tangents)
    d = np.asarray(sf.pairwise_distance(pts, alpha=alpha))
    assert np.all(np.isfinite(d))
    for i, j, k in np.random.default_rng(4).choice(len(pts), size=(30, 3)):
        assert d[i, k] <= d[i, j] + d[j, k] + 1e-10


# ---------------------------------------------------------------- review findings: metric for extreme alpha


def test_inner_without_cancellation_for_extreme_alpha():
    """g(U E, U E) = |E|^2 / (2 (1 + alpha)). The form with a subtraction returned 0 at alpha = 1e16."""
    u = jnp.eye(2)
    d = jnp.array([[0.0, -1.0], [1.0, 0.0]])
    for alpha in (1e16, 1e8, -1 + 1e-12):
        np.testing.assert_allclose(sf.inner(u, d, d, alpha=alpha), 1 / (1 + alpha), rtol=1e-12)
    d32 = d.astype(jnp.float32)
    np.testing.assert_allclose(sf.inner(u.astype(jnp.float32), d32, d32, alpha=1e8), 1e-8, rtol=1e-6)
    # It equals the formula of the module docstring for moderate alpha, also for non-tangent arguments.
    x = sf.random_point(jax.random.key(73), 8, 3)
    a, b = jax.random.normal(jax.random.key(74), (8, 3)), jax.random.normal(jax.random.key(75), (8, 3))
    for alpha in ALPHAS + [5.0]:
        c = (2 * alpha + 1) / (2 * (alpha + 1))
        dense = jnp.sum(a * b) - c * jnp.sum((x.T @ a) * (x.T @ b))
        np.testing.assert_allclose(sf.inner(x, a, b, alpha=alpha), dense, rtol=1e-13, atol=1e-14)


# ---------------------------------------------------------------- second review: domain of exp, derivative


def _tangent_with_exponents(seed, n, p, horizontal, normal, alpha):
    """A tangent with ``|A|_F = horizontal`` and ``|K|_F = normal``, the quantities that enter the
    exponentials of exp."""
    k1, k2, k3 = jax.random.split(jax.random.key(seed), 3)
    base = sf.random_point(k1, n, p)
    g = jax.random.normal(k2, (p, p))
    a = (g - g.T) if p > 1 else jnp.zeros((1, 1))
    a = a * (horizontal / jnp.linalg.norm(a)) if p > 1 else a
    z = sf.project_tangent(base, jax.random.normal(k3, (n, p)))
    z = z - base @ (base.T @ z)
    return base, base @ a + z * (normal / jnp.linalg.norm(z))


@pytest.mark.parametrize("alpha", [0.0, -0.5, 1.0, -0.9, 3.0, 1e16])
def test_exp_s1_in_the_domain_of_its_exponents(alpha):
    """S1 holds where |A| / (1 + alpha) <= 1000, |alpha| |A| / (1 + alpha) <= 1000 and |K| <= 500.

    At the corner of that domain, at most 0.84 of the bound was measured.
    """
    horizontal = 1000 * (1 + alpha) / max(1.0, abs(alpha))
    for shape in [(8, 1), (10, 3), (6, 6), (12, 4), (20, 8)]:
        for seed in range(2):
            base, d = _tangent_with_exponents(seed, *shape, horizontal, 500.0, alpha)
            assert frame_error(sf.exp(base, d, alpha=alpha)) <= s1_bound(shape[1]), (shape, seed)


def test_exp_rejects_alpha_outside_the_metric_family():
    """alpha <= -1 and a non-finite alpha raise an error.

    The reviewer's case, alpha = 1e16 with a tangent of norm 1, lies outside the stated domain
    (|alpha| |A| / (1 + alpha) = 1.4e8). There, the result is NaN and not a frame.
    """
    base = jnp.eye(2)
    j = jnp.array([[0.0, -1.0], [1.0, 0.0]])
    for alpha in (-1.0, -2.0, float("nan"), float("inf")):
        with pytest.raises(ValueError, match="alpha"):
            sf.exp(base, j, alpha=alpha)
    np.testing.assert_allclose(sf.norm(base, 1e8 * j, alpha=1e16), 1.0)
    assert np.all(np.isnan(sf.exp(base, 1e8 * j, alpha=1e16)))


@pytest.mark.parametrize("alpha", ALPHAS)
@pytest.mark.parametrize("length", [1e-8, 1.0, 30.0, 300.0])
def test_exp_derivatives_at_long_tangents(alpha, length):
    """Forward and reverse mode against central differences.

    With the block K^T K of the old form of the derivative, the JVP was NaN at |K| = 1000. The balanced block
    has entries of the order of |K|.
    """
    h = 1e-4
    for shape in [(9, 3), (5, 3), (6, 1)]:
        base = sf.random_point(jax.random.key(80), *shape)
        start = sf.random_tangent(jax.random.key(81), base, alpha=alpha)
        start = length * start / sf.norm(base, start, alpha=alpha)
        direction = sf.random_tangent(jax.random.key(82), base, alpha=alpha)
        direction = direction / sf.norm(base, direction, alpha=alpha)
        weights = jax.random.normal(jax.random.key(83), shape)

        def curve(s, base=base, start=start, direction=direction):
            return sf.exp(base, start + s * direction, alpha=alpha)

        forward = jax.jvp(curve, (0.0,), (1.0,))[1]
        central = (curve(h) - curve(-h)) / (2 * h)
        np.testing.assert_allclose(forward, central, atol=1e-8)
        reverse = jax.grad(
            lambda d, base=base, weights=weights: jnp.sum(sf.exp(base, d, alpha=alpha) * weights)
        )(start)
        np.testing.assert_allclose(jnp.sum(reverse * direction), jnp.sum(central * weights), atol=1e-8)


def test_exp_derivative_of_a_long_normal_component():
    """The reviewer's case: alpha = 0, base e1 and tangent 1000 e2 had a finite value and NaN derivatives."""
    base = jnp.array([[1.0], [0.0]])
    d = jnp.array([[0.0], [1000.0]])
    e = jnp.array([[0.0], [1.0]])
    forward = jax.jvp(lambda x: sf.exp(base, x), (d,), (e,))[1]
    np.testing.assert_allclose(
        forward, (sf.exp(base, d + 1e-4 * e) - sf.exp(base, d - 1e-4 * e)) / 2e-4, atol=1e-8
    )
    # The derivative along the geodesic is the velocity, (-sin 1000, cos 1000).
    np.testing.assert_allclose(forward.ravel(), [-np.sin(1000.0), np.cos(1000.0)], atol=1e-12)
    reverse = jax.grad(lambda x: jnp.sum(sf.exp(base, x)))(d)
    assert np.all(np.isfinite(reverse))
    np.testing.assert_allclose(jnp.sum(reverse * e), jnp.sum(forward), atol=1e-12)


# ---------------------------------------------------------------- second review: residual for inexact frames


def test_log_residual_for_a_frame_with_an_accepted_defect():
    """U = sqrt(1 + 9000 eps) I_100 has defect 9000 eps, within the bound 1e4 eps for tol = 100 eps.

    The residual of the coordinates was 2.83e-11, above the endpoint bound 2.22e-11, for an endpoint error of
    0.
    """
    eps = np.finfo(np.float64).eps
    u = jnp.sqrt(1 + 9000 * eps) * jnp.eye(100)
    res = sf.log(u, u, tol=100 * eps)
    assert res.converged and float(res.residual) <= 1e-14
    np.testing.assert_array_equal(res.tangent, 0.0)
    assert float(sf.distance(u, u, tol=100 * eps)) == 0.0


@pytest.mark.parametrize("shape", [(100, 100), (40, 20), (10, 3), (6, 1)])
def test_log_of_random_inexact_frames(shape):
    """log(U, U) converges to the zero tangent for frames just inside the accepted defect.

    The residual of log(U, Y) is the endpoint error of the public exp to within 5 percent.
    """
    n, p = shape
    eps = np.finfo(np.float64).eps
    for seed in range(2):
        f = sf.random_point(jax.random.key(90 + seed), n, p)
        for tol in (None, 100 * eps):
            cutoff = max(min(1e-12 if tol is None else tol, np.sqrt(eps)), 100 * p * eps)
            b, d = defective(f, 0.95 * cutoff, 92 + seed)
            assert d <= cutoff
            res = sf.log(b, b, tol=tol)
            assert res.converged, (shape, seed, tol, res.residual)
            assert float(jnp.max(jnp.abs(res.tangent))) <= 1e-13
        t = sf.random_tangent(jax.random.key(94 + seed), f)
        y = sf.exp(f, 0.7 * t / sf.norm(f, t))
        res = sf.log(b, y)
        assert res.converged
        endpoint = float(jnp.linalg.norm(sf.exp(b, res.tangent) - y))
        np.testing.assert_allclose(float(res.residual), endpoint, rtol=0.05, atol=1e-14)
