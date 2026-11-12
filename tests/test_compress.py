import jax
import jax.numpy as jnp
import numpy as np
import pytest

import stiefel as sf

EPS = np.finfo(np.float64).eps


def frames_in_subspace(seed, n, dim, m, p, spread=0.5):
    """``m`` frames of St(n, p) that lie in a random ``dim``-dimensional subspace and near one another."""
    keys = jax.random.split(jax.random.key(seed), m + 2)
    w = sf.random_point(keys[0], n, dim)
    centre = sf.random_point(keys[1], dim, p)
    small = []
    for k in keys[2:]:
        t = sf.random_tangent(k, centre)
        small.append(sf.exp(centre, spread * t / sf.norm(centre, t)))
    return w, jnp.stack(small)


def check_basis(frames, jb):
    k = jb.basis.shape[1]
    np.testing.assert_allclose(jb.basis.T @ jb.basis, np.eye(k), atol=100 * k * EPS)
    for f, c in zip(frames, jb.coordinates, strict=True):
        assert sf.is_frame(c)
        assert np.max(np.abs(jb.basis @ c - f)) <= 1e3 * EPS
    np.testing.assert_allclose(sf.lift(jb.basis, jb.coordinates), frames, atol=1e3 * EPS)


def test_joint_basis_full_span():
    w, small = frames_in_subspace(0, 300, 40, 5, 4)
    frames = sf.lift(w, small)
    jb = sf.joint_basis(frames)
    assert jb.basis.shape == (300, 20) and jb.coordinates.shape == (5, 20, 4)
    check_basis(frames, jb)


def test_joint_basis_rank_deficient_span():
    """Twelve frames in a 7-dimensional subspace: K is the rank, which is at least 2p."""
    w, small = frames_in_subspace(1, 200, 7, 12, 3)
    frames = sf.lift(w, small)
    jb = sf.joint_basis(frames)
    assert jb.basis.shape[1] == 7
    check_basis(frames, jb)


@pytest.mark.parametrize("m", [1, 3])
def test_joint_basis_completes_to_2p(m):
    """Identical frames span p dimensions; the basis is completed to 2p orthonormal directions."""
    f = sf.random_point(jax.random.key(2), 50, 4)
    frames = jnp.stack([f] * m)
    jb = sf.joint_basis(frames)
    assert jb.basis.shape == (50, 8)
    check_basis(frames, jb)


def test_joint_basis_identity_when_mp_at_least_n():
    frames = jnp.stack([sf.random_point(k, 10, 3) for k in jax.random.split(jax.random.key(3), 4)])
    jb = sf.joint_basis(frames)
    np.testing.assert_array_equal(jb.basis, np.eye(10))
    np.testing.assert_array_equal(jb.coordinates, frames)


def test_lift_batch_axes():
    w, small = frames_in_subspace(4, 30, 8, 6, 2)
    batched = small.reshape(2, 3, 8, 2)
    assert sf.lift(w, batched).shape == (2, 3, 30, 2)
    np.testing.assert_allclose(sf.lift(w, small[0]), w @ small[0])


@pytest.mark.parametrize("alpha", [0.0, -0.5])
def test_s3_compression_is_exact(alpha):
    """Ambient and compressed exp, log and distance agree for n = 5000, p = 6."""
    n, p = 5000, 6
    w, small = frames_in_subspace(5, n, 24, 4, p, spread=0.8)
    frames = sf.lift(w, small)
    jb = sf.joint_basis(frames)
    c = jb.coordinates
    for i, j in [(0, 1), (2, 3), (1, 3)]:
        ambient = sf.log(frames[i], frames[j], alpha=alpha)
        small_log = sf.log(c[i], c[j], alpha=alpha)
        assert ambient.converged and small_log.converged
        np.testing.assert_allclose(sf.lift(jb.basis, small_log.tangent), ambient.tangent, atol=1e-10)
        np.testing.assert_allclose(
            sf.distance(frames[i], frames[j], alpha=alpha), sf.distance(c[i], c[j], alpha=alpha), atol=1e-10
        )
        np.testing.assert_allclose(
            sf.exp(frames[i], ambient.tangent, alpha=alpha),
            sf.lift(jb.basis, sf.exp(c[i], small_log.tangent, alpha=alpha)),
            atol=1e-10,
        )


def assert_reconstruction(frames, jb, bound=1e3 * EPS):
    k = jb.basis.shape[1]
    assert np.all(np.isfinite(jb.basis))
    assert np.max(np.abs(jb.basis.T @ jb.basis - np.eye(k))) <= 100 * k * EPS
    assert np.max(np.abs(sf.lift(jb.basis, jb.coordinates) - frames)) <= bound
    gram = jnp.einsum("mkp,mkq->mpq", jb.coordinates, jb.coordinates)
    assert np.max(np.abs(gram - np.eye(frames.shape[2]))) <= 100 * frames.shape[2] * EPS


def test_joint_basis_many_duplicates_and_near_duplicates():
    """W1: 200 copies of one frame of St(4000, 5), exact and with one copy perturbed by 1e-12 or 1e-10.

    The QR and SVD routes of LAPACK returned NaN here. The basis must be finite, of size 2p, and exact.
    """
    n, p, m = 4000, 5, 200
    f = sf.random_point(jax.random.key(6), n, p)
    e = jax.random.normal(jax.random.key(7), (n,))
    e = e - f @ (f.T @ e)
    e = e / jnp.linalg.norm(e)
    copies = jnp.broadcast_to(f, (m, n, p))
    for delta in (0.0, 1e-12, 1e-10):
        frames = copies.at[-1].set(sf.polar(f.at[:, 0].add(delta * e))) if delta else copies
        jb = sf.joint_basis(frames)
        assert jb.basis.shape[1] == 2 * p
        assert_reconstruction(frames, jb)


def test_joint_basis_default_rtol_keeps_a_small_direction():
    """C2: one frame perturbed by 1e-12 among 99 copies of three frames of St(1100, 10).

    The old relative threshold dropped that direction and broke the 1e3 eps bound. The default threshold
    keeps the direction.
    """
    n, p = 1100, 10
    f0, f1, f2 = (sf.random_point(jax.random.key(k), n, p) for k in (8, 9, 10))
    e = jnp.zeros(n).at[3].set(1.0)
    e = e - f0 @ (f0.T @ e)
    e = e / jnp.linalg.norm(e)
    last = sf.polar(f0.at[:, 0].add(1e-12 * e))
    frames = jnp.concatenate([jnp.stack([f0, f1, f2] * 33), last[None]])
    jb = sf.joint_basis(frames)
    assert jb.basis.shape[1] == 3 * p + 1
    assert_reconstruction(frames, jb)
    np.testing.assert_allclose(sf.distance(jb.coordinates[0], jb.coordinates[-1]), 1e-12, rtol=1e-3)
    # A loose rtol drops the direction, within its stated bound rtol + 1e3 eps.
    loose = sf.joint_basis(frames, rtol=1e-9)
    assert loose.basis.shape[1] == 3 * p
    assert_reconstruction(frames, loose, bound=1e-9 + 1e3 * EPS)


def test_joint_basis_generic_frames_and_validation():
    frames = jnp.stack([sf.random_point(k, 200, 3) for k in jax.random.split(jax.random.key(11), 20)])
    jb = sf.joint_basis(frames)
    assert jb.basis.shape == (200, 60)
    assert_reconstruction(frames, jb)
    for bad in (-1.0, float("nan"), float("inf")):
        with pytest.raises(ValueError):
            sf.joint_basis(frames, rtol=bad)
    with pytest.raises(ValueError):
        sf.joint_basis(frames.at[0, 0, 0].set(jnp.nan))


def test_factorized_route_agrees_with_the_gram_schmidt_route():
    # The fast route (one QR and a small SVD) and the column-by-column route must span the same space.
    from stiefel import compress

    key = jax.random.PRNGKey(11)
    frames = jax.vmap(lambda k: sf.random_point(k, 300, 4))(jax.random.split(key, 9))
    fast = compress._factorized_basis(frames, 500 * float(jnp.finfo(frames.dtype).eps))
    assert fast is not None
    original = compress._factorized_basis
    try:
        compress._factorized_basis = lambda frames, threshold: None
        slow = compress.joint_basis(frames)
    finally:
        compress._factorized_basis = original
    assert fast.basis.shape == slow.basis.shape
    probe = jax.random.normal(jax.random.PRNGKey(12), (300, 5))
    projected_fast = fast.basis @ (fast.basis.T @ probe)
    projected_slow = slow.basis @ (slow.basis.T @ probe)
    assert float(jnp.max(jnp.abs(projected_fast - projected_slow))) < 1e-12
    for jb in (fast, slow):
        assert float(jnp.max(jnp.abs(sf.lift(jb.basis, jb.coordinates) - frames))) < 1e3 * 2.3e-16


# ---------------------------------------------------------------- second review: one contract on both routes


def _gram_schmidt_route(frames, **kwargs):
    from stiefel import compress

    original = compress._factorized_basis
    try:
        compress._factorized_basis = lambda frames, threshold: None
        return compress.joint_basis(frames, **kwargs)
    finally:
        compress._factorized_basis = original


def test_joint_basis_coordinates_are_frames_for_a_loose_rtol():
    """St(4, 1): e1, e2 and cos(0.05) e1 + sin(0.05) e3 with rtol = 0.1.

    The Gram-Schmidt route returned the third coordinate with squared norm 0.9975. Both routes must return
    frames within rtol (1 + p rtol).
    """
    e = jnp.eye(4)
    frames = jnp.stack([e[:, :1], e[:, 1:2], jnp.cos(0.05) * e[:, :1] + jnp.sin(0.05) * e[:, 2:3]])
    rtol = 0.1
    for jb in (sf.joint_basis(frames, rtol=rtol), _gram_schmidt_route(frames, rtol=rtol)):
        assert jb.basis.shape == (4, 2)
        assert_reconstruction(frames, jb, bound=rtol * (1 + rtol) + 1e3 * EPS)
        # The polar factor of the projection is the nearest frame. The error is the dropped part, 0.05.
        assert np.max(np.abs(sf.lift(jb.basis, jb.coordinates) - frames)) <= 0.051


@pytest.mark.parametrize("rtol", [None, 1e-9, 1e-3])
def test_joint_basis_routes_share_the_contract(rtol):
    """The default-rtol cases and two looser thresholds on both routes: near-duplicates, a rank-deficient span
    and generic frames."""
    f = sf.random_point(jax.random.key(13), 60, 3)
    e = jnp.zeros(60).at[5].set(1.0)
    e = e - f @ (f.T @ e)
    e = e / jnp.linalg.norm(e)
    near = jnp.stack([f, f, sf.polar(f.at[:, 0].add(1e-6 * e))])
    w, small = frames_in_subspace(14, 60, 7, 5, 3)
    generic = jnp.stack([sf.random_point(k, 60, 3) for k in jax.random.split(jax.random.key(15), 6)])
    t = 500 * EPS if rtol is None else rtol
    for frames in (near, sf.lift(w, small), generic):
        for jb in (sf.joint_basis(frames, rtol=rtol), _gram_schmidt_route(frames, rtol=rtol)):
            assert_reconstruction(frames, jb, bound=t * (1 + 3 * t) + 1e3 * EPS)


def test_joint_basis_gram_schmidt_route_raises_on_a_failed_verification():
    from stiefel import compress

    frames = jnp.stack([sf.random_point(k, 30, 2) for k in jax.random.split(jax.random.key(16), 3)])
    original = compress._extend
    try:
        compress._extend = lambda *args: (lambda b, k: (b.at[0, 0].set(jnp.nan), k))(*original(*args))
        with pytest.raises(FloatingPointError, match="verification"):
            _gram_schmidt_route(frames)
    finally:
        compress._extend = original
