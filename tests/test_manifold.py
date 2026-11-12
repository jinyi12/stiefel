import jax
import jax.numpy as jnp
import numpy as np
import pytest

import stiefel as sf


def test_constructor_validation_and_dim():
    assert sf.Stiefel(7, 3).dim == 7 * 3 - 6
    for bad in [dict(n=2, p=3), dict(n=3, p=0)]:
        with pytest.raises(ValueError):
            sf.Stiefel(**bad)
    with pytest.raises(ValueError):
        sf.Stiefel(4, 2, alpha=-1.0)


@pytest.mark.parametrize("alpha", [0.0, -0.5, 1.0])
def test_methods_forward_and_gradient(alpha):
    m = sf.Stiefel(9, 3, alpha=alpha)
    x = m.random_point(jax.random.key(0))
    assert sf.is_frame(x) and sf.is_frame(m.retraction(x, m.random_tangent_vector(jax.random.key(1), x)))
    d = m.random_tangent_vector(jax.random.key(2), x)
    d = 0.5 * d / m.norm(x, d)
    y = m.exp(x, d)
    np.testing.assert_allclose(m.log(x, y), d, atol=1e-10)
    np.testing.assert_allclose(m.dist(x, y), 0.5, rtol=1e-10)
    assert m.log_result(x, y).converged
    np.testing.assert_array_equal(m.zero_vector(x), np.zeros((9, 3)))
    np.testing.assert_allclose(m.transport(x, y, d), sf.project_tangent(y, d))
    # The Riemannian gradient represents the Frobenius product on the tangent space.
    e = jax.random.normal(jax.random.key(3), (9, 3))
    g = m.euclidean_to_riemannian_gradient(x, e)
    for k in range(4, 7):
        t = m.random_tangent_vector(jax.random.key(k), x)
        np.testing.assert_allclose(m.inner_product(x, g, t), jnp.sum(e * t), rtol=1e-12)
    s = x.T @ g
    np.testing.assert_allclose(s, -s.T, atol=1e-14)


def test_log_is_nan_without_convergence():
    m = sf.Stiefel(6, 2)
    x = m.random_point(jax.random.key(7))
    assert np.all(np.isnan(m.log(x, x.at[:, 0].multiply(-1.0), max_iter=20)))
    assert np.isnan(m.dist(x, x.at[:, 0].multiply(-1.0), max_iter=20))


# ---------------------------------------------------------------- S9: pymanopt (alpha = -1/2)


def test_agrees_with_pymanopt():
    pymanopt = pytest.importorskip("pymanopt")
    n, p = 10, 3
    theirs = pymanopt.manifolds.Stiefel(n, p)
    ours = sf.Stiefel(n, p, alpha=-0.5)
    x = ours.random_point(jax.random.key(8))
    v = jax.random.normal(jax.random.key(9), (n, p))
    a, b = ours.projection(x, v), ours.random_tangent_vector(jax.random.key(10), x)
    xn, an, bn = (np.asarray(t) for t in (x, a, b))
    np.testing.assert_allclose(ours.projection(x, v), theirs.projection(xn, np.asarray(v)), atol=1e-8)
    np.testing.assert_allclose(ours.inner_product(x, a, b), theirs.inner_product(xn, an, bn), atol=1e-8)
    np.testing.assert_allclose(ours.exp(x, a), theirs.exp(xn, an), atol=1e-8)
    np.testing.assert_allclose(ours.retraction(x, a), theirs.retraction(xn, an), atol=1e-8)


def test_pymanopt_adapter_dominant_subspace():
    pymanopt = pytest.importorskip("pymanopt")
    n, p = 12, 3
    rng = np.random.default_rng(0)
    q = np.linalg.qr(rng.normal(size=(n, n)))[0]
    eigvals = np.linspace(10.0, 1.0, n)
    a = (q * eigvals) @ q.T
    manifold = sf.Stiefel(n, p).to_pymanopt(jax.random.key(11))

    @pymanopt.function.numpy(manifold)
    def cost(x):
        return -np.trace(x.T @ a @ x)

    @pymanopt.function.numpy(manifold)
    def gradient(x):
        return -2 * a @ x

    problem = pymanopt.Problem(manifold, cost, euclidean_gradient=gradient)
    result = pymanopt.optimizers.ConjugateGradient(verbosity=0, max_iterations=500).run(problem)
    x = result.point
    top = q[:, :p]
    np.testing.assert_allclose(x @ x.T, top @ top.T, atol=1e-6)
    np.testing.assert_allclose(cost(x), -eigvals[:p].sum(), rtol=1e-10)


# ---------------------------------------------------------------- S9: geomstats (alpha = 0)


def test_agrees_with_geomstats():
    # geomstats 2.8 imports numpy.trapz, which numpy 2 removed. That ImportError also means "unavailable".
    stiefel_module = pytest.importorskip("geomstats.geometry.stiefel", exc_type=ImportError)
    n, p = 8, 3
    space = stiefel_module.Stiefel(n, p)
    x = sf.random_point(jax.random.key(12), n, p)
    a = sf.random_tangent(jax.random.key(13), x)
    b = sf.random_tangent(jax.random.key(14), x)
    a = 0.7 * a / sf.norm(x, a)
    xn, an, bn = (np.asarray(t) for t in (x, a, b))
    y = sf.exp(x, a)
    np.testing.assert_allclose(y, space.metric.exp(an, xn), atol=1e-8)
    np.testing.assert_allclose(sf.log(x, y).tangent, space.metric.log(np.asarray(y), xn), atol=1e-8)
    np.testing.assert_allclose(sf.inner(x, a, b), space.metric.inner_product(an, bn, xn), atol=1e-8)
