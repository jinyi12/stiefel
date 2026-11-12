"""Intrinsic objective, numerical failure reports, pruning and mass-normalized mini-batch updates."""

import jax
import jax.numpy as jnp
import numpy as np
import pytest

import stiefel as sf
from stiefel.statistics import _ema_directions, _nearest_centres


@pytest.fixture(autouse=True)
def _free_compiled_code():
    # The nested assignment loops otherwise exhaust vm.max_map_count across the metric parametrizations.
    yield
    jax.clear_caches()


def clusters(alpha=0.0):
    base = sf.random_point(jax.random.key(101), 12, 2)
    keys = jax.random.split(jax.random.key(102), 3)
    centres = []
    points = []
    for i, key in enumerate(keys):
        tangent = sf.random_tangent(key, base, alpha=alpha)
        centre = sf.exp(base, 0.7 * tangent / sf.norm(base, tangent, alpha=alpha), alpha=alpha)
        centres.append(centre)
        noise = jax.vmap(lambda k, c=centre: sf.random_tangent(k, c, alpha=alpha))(
            jax.random.split(jax.random.key(103 + i), 6)
        )
        noise -= noise.mean(axis=0)
        noise *= 0.04 / jax.vmap(lambda v, c=centre: sf.norm(c, v, alpha=alpha))(noise).max()
        points.append(jax.vmap(lambda v, c=centre: sf.exp(c, v, alpha=alpha))(noise))
    return jnp.stack(centres), jnp.concatenate(points)


def test_mass_normalization_reduces_to_euclidean_ema():
    centres = jnp.array([2.0, -3.0, 1.0])
    masses = jnp.array([4.0, 2.0, 3.0])
    points = jnp.array([1.0, 4.0, -2.0])
    labels = jnp.array([0, 0, 1])
    step = 0.2
    directions, updated = _ema_directions(labels, (points - centres[labels])[:, None, None], masses, step)
    sums = jax.ops.segment_sum(points, labels, 3)
    expected = ((1 - step) * masses * centres + step * sums) / updated
    np.testing.assert_allclose(centres + directions[:, 0, 0], expected, atol=1e-14)
    assert directions[2, 0, 0] == 0
    zero, updated = _ema_directions(labels, (points - centres[labels])[:, None, None], masses * 0, 1.0)
    assert jnp.all(jnp.isfinite(zero)) and updated[2] == 0


@pytest.mark.parametrize("alpha", [-0.5, 0.0, 1.0])
def test_pruned_assignments_match_exhaustive_geodesic_distances(alpha):
    centres, points = clusters(alpha)
    labels, distances, tangents, valid, calls = _nearest_centres(centres, points, alpha, 100)
    exhaustive = jax.vmap(lambda x: jax.vmap(lambda c: sf.distance(c, x, alpha=alpha))(centres))(points)
    np.testing.assert_array_equal(labels, jnp.argmin(exhaustive, axis=1))
    np.testing.assert_allclose(distances, exhaustive.min(axis=1), atol=1e-10)
    rebuilt = jax.vmap(lambda c, v: sf.exp(c, v, alpha=alpha))(centres[labels], tangents)
    np.testing.assert_allclose(rebuilt, points, atol=1e-10)
    assert valid.all() and calls.sum() < len(points) * len(centres)


@pytest.mark.parametrize("alpha", [-0.5, 0.0, 1.0])
def test_emeans_recovers_clusters_and_intrinsic_means(alpha):
    centres, points = clusters(alpha)
    result = sf.emeans(jax.random.key(104), points, 3, alpha=alpha, epochs=3, batch_size=7, tol=1e-7)
    assert result.converged
    labels = np.asarray(result.labels)
    assert len(set(labels)) == 3
    assert all(len(set(labels[i : i + 6])) == 1 for i in range(0, 18, 6))
    for i in range(3):
        assert sf.is_frame(result.centers[i])
        members = points[labels == i]
        exact = sf.frechet_mean(members, alpha=alpha)
        np.testing.assert_allclose(result.centers[i], exact.mean, atol=3e-7)
    own = jax.vmap(lambda c, x: sf.distance(c, x, alpha=alpha))(result.centers[result.labels], points)
    np.testing.assert_allclose(result.inertia, jnp.sum(own**2), atol=1e-10)
    repeated = sf.emeans(jax.random.key(104), points, 3, alpha=alpha, epochs=3, batch_size=7, tol=1e-7)
    np.testing.assert_array_equal(result.labels, repeated.labels)
    np.testing.assert_array_equal(result.centers, repeated.centers)


def test_emeans_circle_mean_and_unfinished_refinement():
    angles = jnp.array([-0.5, -0.1, 0.2, 0.7])
    points = jnp.stack([jnp.cos(angles), jnp.sin(angles)], axis=1)[..., None]
    result = sf.emeans(jax.random.key(105), points, 1, epochs=2, batch_size=2)
    expected = jnp.array([[jnp.cos(angles.mean())], [jnp.sin(angles.mean())]])
    assert result.converged
    np.testing.assert_allclose(result.centers[0], expected, atol=1e-10)
    unfinished = sf.emeans(jax.random.key(105), points, 1, epochs=1, batch_size=1, polish_iter=0)
    assert not unfinished.converged and jnp.isfinite(unfinished.inertia)


def test_assignment_uses_intrinsic_distance_when_chordal_order_disagrees():
    point = jnp.eye(4, 2)
    rotation = point @ jnp.array([[0.0, -0.8], [0.8, 0.0]])
    normal = jnp.zeros_like(point).at[2, 0].set(0.95)
    centres = jnp.stack([sf.exp(point, rotation), sf.exp(point, normal)])
    assert jnp.argmin(jnp.sum((centres - point) ** 2, axis=(1, 2))) == 1
    labels, distances, _, valid, calls = _nearest_centres(centres, point[None], 0.0, 100)
    assert valid[0] and labels[0] == 0 and calls[0] == 2
    np.testing.assert_allclose(distances[0], 0.8, atol=1e-10)


def test_emeans_duplicates_and_one_cluster_per_point():
    _, points = clusters()
    for data, k in [(points[:4], 4), (jnp.repeat(points[:1], 4, axis=0), 3)]:
        result = sf.emeans(jax.random.key(106), data, k, epochs=1)
        assert result.converged and result.inertia < 1e-20
        assert set(np.asarray(result.labels)) == set(range(k))
        assert jax.vmap(sf.is_frame)(result.centers).all()


def test_emeans_reports_failed_logarithms():
    # Opposite components of O(2) cannot share an intrinsic centre with finite distances to both points.
    point = sf.random_point(jax.random.key(107), 2, 2)
    points = jnp.stack([point, point.at[:, 0].multiply(-1)])
    result = sf.emeans(jax.random.key(108), points, 1, epochs=1, log_max_iter=1)
    assert not result.converged and jnp.isnan(result.inertia)
    assert sf.is_frame(result.centers[0])


def test_emeans_float32_and_compression():
    _, points = clusters()
    result32 = sf.emeans(jax.random.key(109), points.astype(jnp.float32), 3, epochs=2)
    assert result32.centers.dtype == jnp.float32 and result32.inertia.dtype == jnp.float32
    assert result32.converged
    basis = sf.random_point(jax.random.key(110), 200, 12)
    small = sf.emeans(jax.random.key(109), points, 3, epochs=2)
    large = sf.emeans(jax.random.key(109), sf.lift(basis, points), 3, epochs=2)
    np.testing.assert_array_equal(large.labels, small.labels)
    np.testing.assert_allclose(large.centers, sf.lift(basis, small.centers), atol=1e-9)
    np.testing.assert_allclose(large.inertia, small.inertia, atol=1e-10)


@pytest.mark.parametrize(
    "bad",
    [
        {"n_clusters": 0},
        {"n_clusters": 20},
        {"epochs": 0},
        {"batch_size": 0},
        {"batch_size": 1.5},
        {"decay": 1.0},
        {"decay": jnp.nan},
        {"polish_iter": -1},
        {"alpha": -1.0},
        {"tol": 0.0},
        {"log_max_iter": 0},
    ],
)
def test_emeans_validation(bad):
    point = sf.random_point(jax.random.key(111), 8, 2)
    options = {"n_clusters": 1, **bad}
    with pytest.raises((ValueError, TypeError)):
        sf.emeans(jax.random.key(112), point[None], **options)


def test_emeans_rejects_invalid_frames_and_differentiation():
    point = sf.random_point(jax.random.key(113), 8, 2)
    for invalid in [point[None] * 2, point[None].at[0, 0, 0].set(jnp.nan)]:
        with pytest.raises(ValueError):
            sf.emeans(jax.random.key(114), invalid, 1)
    with pytest.raises(TypeError, match="not differentiable"):
        jax.grad(lambda x: sf.emeans(jax.random.key(114), x, 1).inertia)(point[None])


def test_mean_solvers_do_not_evaluate_zero_memberships(monkeypatch):
    import stiefel.statistics as stats

    seen = []
    original = stats._frechet_loop

    def record(points, weights, *args):
        seen.append(len(points))
        assert jnp.all(weights > 0)
        return original(points, weights, *args)

    monkeypatch.setattr(stats, "_frechet_loop", record)
    _, points = clusters()
    mean = sf.frechet_mean(points, weights=jnp.eye(len(points))[0])
    assert mean.converged and seen == [1]
    seen.clear()
    result = sf.kmeans(jax.random.key(115), points, 3)
    assert result.converged
    assert sum(seen) == len(points) * (int(result.iterations) - 1)


def test_emeans_retains_valid_centres_if_an_update_fails(monkeypatch):
    import stiefel.statistics as stats

    point = sf.random_point(jax.random.key(116), 8, 2)
    monkeypatch.setattr(stats, "exp", lambda c, v, alpha: jnp.full_like(c, jnp.nan))
    centre, _, _, valid, _ = stats._emeans_batch(point[None], jnp.ones(1), point[None], 0.1, 0.0, 100)
    assert not valid
    np.testing.assert_array_equal(centre, point[None])


def test_emeans_retains_valid_centres_if_refinement_fails(monkeypatch):
    import stiefel.statistics as stats

    def fail(points, weights, init, *args):
        return jnp.full_like(init, jnp.nan), jnp.asarray(jnp.nan), jnp.asarray(0), jnp.asarray(False)

    monkeypatch.setattr(stats, "_frechet_loop", fail)
    point = sf.random_point(jax.random.key(117), 8, 2)
    result = sf.emeans(jax.random.key(118), point[None], 1, epochs=1)
    assert not result.converged and jnp.isfinite(result.inertia)
    assert sf.is_frame(result.centers[0])


def test_emeans_uses_distinct_random_keys():
    point = sf.random_point(jax.random.key(119), 8, 2)
    with jax.debug_key_reuse(True):
        result = sf.emeans(jax.random.key(120), point[None], 1, epochs=1)
    assert result.converged
