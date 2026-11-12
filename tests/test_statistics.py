import jax
import jax.numpy as jnp
import numpy as np
import pytest

import stiefel as sf

EPS = np.finfo(np.float64).eps


def scatter(seed, centre, m, radius, alpha=0.0, zero_mean=False):
    """``m`` points ``exp(centre, T_i)`` with tangents of norm ``radius``, or of zero mean and largest norm
    ``radius``."""
    keys = jax.random.split(jax.random.key(seed), m)
    t = jnp.stack([sf.random_tangent(k, centre, alpha=alpha) for k in keys])
    norms = jax.vmap(lambda x: sf.norm(centre, x, alpha=alpha))
    if zero_mean:
        t = t - t.mean(axis=0)
        t = t * radius / norms(t).max()
    else:
        t = t * (radius / norms(t))[:, None, None]
    return jax.vmap(lambda x: sf.exp(centre, x, alpha=alpha))(t), t


def three_clusters(seed, n=30, p=3, per=8):
    keys = jax.random.split(jax.random.key(seed), 5)
    x = sf.random_point(keys[0], n, p)
    centres = []
    for k in keys[1:4]:
        t = sf.random_tangent(k, x)
        centres.append(sf.exp(x, 0.6 * t / sf.norm(x, t)))
    points = jnp.concatenate([scatter(seed + i, c, per, 0.05)[0] for i, c in enumerate(centres)])
    truth = np.repeat(np.arange(3), per)
    perm = np.asarray(jax.random.permutation(keys[4], len(truth)))
    return points[perm], truth[perm]


# ---------------------------------------------------------------- Fréchet mean


@pytest.mark.parametrize("alpha", [0.0, -0.5])
@pytest.mark.parametrize("shape", [(40, 3), (5, 3)])
def test_frechet_mean_recovers_base(alpha, shape):
    base = sf.random_point(jax.random.key(0), *shape)
    points, t = scatter(1, base, 7, 0.4, alpha, zero_mean=True)
    np.testing.assert_allclose(t.sum(axis=0), 0.0, atol=1e-13)
    res = sf.frechet_mean(points, alpha=alpha)
    assert res.converged and res.gradient_norm <= 1e-10
    assert sf.is_frame(res.mean)
    np.testing.assert_allclose(res.mean, base, atol=1e-9)


def test_frechet_mean_single_point_and_weights():
    base = sf.random_point(jax.random.key(2), 12, 2)
    res = sf.frechet_mean(base[None])
    assert res.converged
    np.testing.assert_allclose(res.mean, base, atol=1e-14)
    points, _ = scatter(3, base, 4, 0.3)
    weighted = sf.frechet_mean(points, weights=jnp.array([2.0, 0.0, 0.0, 0.0]))
    assert weighted.converged
    np.testing.assert_allclose(weighted.mean, points[0], atol=1e-10)
    # Weights are normalized: a common factor on all weights does not change the mean.
    w = jnp.array([1.0, 2.0, 3.0, 4.0])
    np.testing.assert_allclose(
        sf.frechet_mean(points, weights=w).mean, sf.frechet_mean(points, weights=10 * w).mean
    )


def test_frechet_mean_with_init_outside_span():
    base = sf.random_point(jax.random.key(4), 30, 2)
    points, _ = scatter(5, base, 3, 0.3, zero_mean=True)
    t = sf.random_tangent(jax.random.key(6), base)
    init = sf.exp(base, 0.2 * t / sf.norm(base, t))
    res = sf.frechet_mean(points, init=init)
    assert res.converged
    np.testing.assert_allclose(res.mean, base, atol=1e-9)


def test_frechet_mean_reports_failed_logarithm():
    base = sf.random_point(jax.random.key(7), 8, 2)
    points = jnp.stack([base, base.at[:, 0].multiply(-1.0)])
    res = sf.frechet_mean(points, init=base, max_iter=5)
    assert not res.converged
    assert sf.is_frame(res.mean)


def test_frechet_mean_tolerance_validation():
    base = sf.random_point(jax.random.key(8), 8, 2)
    with pytest.raises(ValueError):
        sf.frechet_mean(base[None], tol=1e-16)


def test_s3_frechet_mean_compression_is_exact():
    n, p = 5000, 6
    w = sf.random_point(jax.random.key(9), n, 18)
    small_centre = sf.random_point(jax.random.key(10), 18, p)
    small, _ = scatter(11, small_centre, 5, 0.5)
    ambient = sf.frechet_mean(sf.lift(w, small))
    compressed = sf.frechet_mean(small)
    assert ambient.converged and compressed.converged
    np.testing.assert_allclose(ambient.mean, sf.lift(w, compressed.mean), atol=1e-10)
    np.testing.assert_allclose(
        sf.pairwise_distance(sf.lift(w, small)), sf.pairwise_distance(small), atol=1e-10
    )


# ---------------------------------------------------------------- pairwise distances


def test_pairwise_distance():
    base = sf.random_point(jax.random.key(12), 20, 3)
    points, _ = scatter(13, base, 5, 0.5)
    d = sf.pairwise_distance(points)
    assert d.shape == (5, 5)
    np.testing.assert_array_equal(d, d.T)
    np.testing.assert_array_equal(np.diag(d), 0.0)
    for i in range(5):
        for j in range(i + 1, 5):
            np.testing.assert_allclose(d[i, j], sf.distance(points[i], points[j]), atol=1e-12)
    np.testing.assert_array_equal(sf.pairwise_distance(points[:1]), np.zeros((1, 1)))


def test_pairwise_distance_nan_on_failure():
    base = sf.random_point(jax.random.key(14), 8, 2)
    points = jnp.stack([base, base.at[:, 0].multiply(-1.0), base])
    d = sf.pairwise_distance(points, max_iter=20)
    assert np.isnan(d[0, 1]) and np.isnan(d[1, 0])
    assert d[0, 2] <= 1e-12


# ---------------------------------------------------------------- k-means


def same_partition(a, b):
    a, b = np.asarray(a), np.asarray(b)
    return all(len(set(b[a == label])) == 1 for label in set(a)) and len(set(a)) == len(set(b))


def test_kmeans_recovers_three_clusters():
    points, truth = three_clusters(20)
    res = sf.kmeans(jax.random.key(0), points, 3)
    assert res.converged
    assert same_partition(res.labels, truth)
    assert res.centers.shape == (3, 30, 3)
    for c in res.centers:
        assert sf.is_frame(c)
    np.testing.assert_allclose(res.inertia, len(truth) * 0.05**2, rtol=0.3)
    again = sf.kmeans(jax.random.key(0), points, 3)
    np.testing.assert_array_equal(again.labels, res.labels)
    np.testing.assert_array_equal(again.centers, res.centers)
    several = sf.kmeans(jax.random.key(1), points, 3, n_init=3)
    assert several.inertia <= res.inertia * (1 + 1e-9) and same_partition(several.labels, truth)


def test_kmeans_one_cluster_per_point():
    points, _ = three_clusters(21, per=2)
    res = sf.kmeans(jax.random.key(2), points, len(points))
    assert res.inertia <= 1e-20
    assert sorted(np.asarray(res.labels).tolist()) == list(range(len(points)))


def test_kmeans_validation():
    points, _ = three_clusters(22, per=2)
    with pytest.raises(ValueError):
        sf.kmeans(jax.random.key(3), points, 0)
    with pytest.raises(ValueError):
        sf.kmeans(jax.random.key(3), points, 3, tol=1e-16)


# ---------------------------------------------------------------- review findings


def test_frechet_mean_weights_without_overflow_and_validation():
    """Weights of 1e308 overflowed their sum, and the function then returned the first point as the mean."""
    points = jnp.array([[[1.0], [0.0]], [[0.0], [1.0]]])
    res = sf.frechet_mean(points, weights=jnp.array([1e308, 1e308]))
    assert res.converged
    np.testing.assert_allclose(res.mean, np.full((2, 1), np.sqrt(0.5)), atol=1e-12)
    for bad in ([np.inf, 1.0], [np.nan, 1.0], [-1.0, 2.0], [0.0, 0.0]):
        with pytest.raises(ValueError):
            sf.frechet_mean(points, weights=jnp.array(bad))
    with pytest.raises(ValueError, match="non-finite"):
        sf.frechet_mean(points.at[1, 0, 0].set(jnp.nan), weights=jnp.array([1.0, 0.0]))


def test_statistics_float32_at_default_tolerance():
    """U1: float32 input runs at the default tolerances (it raised ValueError) and stays float32 (S4)."""
    points, truth = three_clusters(23, per=3)
    points32 = points.astype(jnp.float32)
    mean = sf.frechet_mean(points32[truth == 0])
    assert mean.converged and mean.mean.dtype == jnp.float32
    res = sf.kmeans(jax.random.key(4), points32, 3)
    assert res.converged and res.centers.dtype == jnp.float32
    assert same_partition(res.labels, truth)
    assert sf.pairwise_distance(points32[:3]).dtype == jnp.float32


def test_log_max_iter_reaches_the_inner_logarithms():
    base = sf.random_point(jax.random.key(15), 10, 3)
    points, _ = scatter(16, base, 3, 1.2, zero_mean=True)
    assert not sf.frechet_mean(points, init=base, log_max_iter=1).converged
    assert sf.frechet_mean(points, init=base, log_max_iter=1000).converged
    assert not sf.kmeans(jax.random.key(5), points, 1, log_max_iter=1).converged


def test_kmeans_reseeds_an_empty_cluster():
    """Two distinct frames and three clusters.

    Two centres coincide, so one cluster is empty after the first assignment and is re-seeded. Every cluster
    ends non-empty.
    """
    x = sf.random_point(jax.random.key(17), 8, 2)
    y = sf.random_point(jax.random.key(18), 8, 2)
    points = jnp.stack([x, y, x, y, x, y])
    for seed in range(3):
        res = sf.kmeans(jax.random.key(seed), points, 3)
        assert sorted(set(np.asarray(res.labels).tolist())) == [0, 1, 2]
        assert res.inertia <= 1e-20 and res.converged
        for c in res.centers:
            assert sf.is_frame(c)


def test_kmeans_reports_a_failed_centre_mean(monkeypatch):
    """converged is False when a Fréchet mean of a centre does not converge, even if all else does."""
    import stiefel.statistics as stats

    original = stats._frechet_loop

    def failing(*args):
        x, g, k, _ = original(*args)
        return x, g, k, jnp.asarray(False)

    points, _ = three_clusters(24, per=2)
    assert sf.kmeans(jax.random.key(6), points, 3).converged
    monkeypatch.setattr(stats, "_frechet_loop", failing)
    assert not sf.kmeans(jax.random.key(6), points, 3).converged
