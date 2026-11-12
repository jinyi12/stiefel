"""Fréchet means, pairwise distances and k-means for frames on the Stiefel manifold.

Each function first compresses its input with :func:`stiefel.compress.joint_basis`, so that the iterations
cost nothing that depends on ``n`` (S3, S6). None of the functions is ``jit``-compatible as a whole, because
the size of the joint basis depends on the data. Their inner loops are compiled.
"""

from functools import partial
from typing import NamedTuple

import jax
import jax.numpy as jnp
import numpy as np
from jaxtyping import Array, Bool, Float, Int, PRNGKeyArray

from stiefel.compress import joint_basis, lift
from stiefel.geometry import _default_tol, distance, exp, log, norm, polar


class FrechetResult(NamedTuple):
    mean: Float[Array, "n p"]
    #: Riemannian norm of the mean tangent vector at the last iterate (zero at a stationary point).
    gradient_norm: Float[Array, ""]
    iterations: Int[Array, ""]
    #: True only if ``gradient_norm <= tol`` and every logarithm that was used had converged.
    converged: Bool[Array, ""]


@partial(jax.jit, static_argnames=("alpha", "tol", "max_iter", "step", "log_max_iter"))
def _frechet_loop(points, weights, init, alpha, tol, max_iter, step, log_max_iter):
    """Fixed-point iteration of :func:`frechet_mean` on compressed frames, with normalized weights.

    Points with zero weight do not enter the mean. A logarithm that fails for one of these points is ignored,
    so that k-means can pass every point with a membership mask.
    """

    def mean_tangent(x):
        res = jax.vmap(lambda y: log(x, y, alpha=alpha, max_iter=log_max_iter))(points)
        used = weights > 0
        g = jnp.einsum("m,mkp->kp", weights, jnp.where(used[:, None, None], res.tangent, 0.0))
        return g, norm(x, g, alpha=alpha), jnp.all(res.converged | ~used)

    def cond(state):
        _, _, g_norm, ok, k = state
        return ok & (g_norm > tol) & (k < max_iter)

    def body(state):
        x, g, _, _, k = state
        x = exp(x, step * g, alpha=alpha)
        return (x, *mean_tangent(x), k + 1)

    x, _, g_norm, ok, k = jax.lax.while_loop(cond, body, (init, *mean_tangent(init), jnp.asarray(0)))
    return x, g_norm, k, ok & (g_norm <= tol)


def frechet_mean(
    points: Float[Array, "M n p"],
    *,
    weights: Float[Array, " M"] | None = None,
    init: Float[Array, "n p"] | None = None,
    alpha: float = 0.0,
    tol: float | None = None,
    max_iter: int = 100,
    step: float = 1.0,
    log_max_iter: int = 100,
) -> FrechetResult:
    """Stationary point of ``sum_i w_i d(X, points[i])^2`` by the fixed-point iteration
    ``X <- exp(X, step * sum_i w_i log(X, points[i]))``.

    Requires:
        THE CALLER SHALL pass frames (S1).
        THE CALLER SHALL NOT read the result as the unique global minimizer, because the iteration is local.
        The exception is a set of points that lie in a geodesic ball in which the mean is unique (Afsari
        2011).
    Ensures:
        IF ``weights`` are given and are not finite, contain a negative value or have a zero sum THEN THE
        CALLABLE SHALL raise ``ValueError``.
        THE CALLABLE SHALL normalize ``weights`` to sum to one, without overflow for weights up to the
        largest float. WHEN ``weights`` is None THEN THE CALLABLE SHALL use equal weights.
        WHEN ``tol`` is None THEN THE CALLABLE SHALL use ``1e-10`` in float64 and ``1e-3`` in float32. A
        non-finite ``tol`` or a ``tol`` below ``100 * eps`` raises ``ValueError`` (S4).
        WHEN ``init`` is None THEN THE CALLABLE SHALL start from ``polar`` of the weighted arithmetic mean.
        THE CALLABLE SHALL return a frame (S1). It SHALL stop when the Riemannian norm of the mean tangent is
        at most ``tol``, or after ``max_iter`` iterations. Each logarithm runs at most ``log_max_iter``
        iterations (the ``max_iter`` of :func:`stiefel.log`). Frames that are far apart can need more than
        the default.
        IF any logarithm fails to converge THEN THE CALLABLE SHALL return ``converged`` False (S2).
        WHEN ``M == 1`` THEN THE CALLABLE SHALL return that point with ``converged`` True.
        IF a transformation differentiates through the callable THEN it SHALL raise ``TypeError`` (S5).
    """
    tol = _default_tol(tol, points.dtype, 1e-10, 1e-3)
    m = points.shape[0]
    if weights is None:
        weights = jnp.full(m, 1.0 / m, points.dtype)
    else:
        w = np.asarray(weights)
        if not (np.all(np.isfinite(w)) and np.all(w >= 0) and np.any(w > 0)):
            raise ValueError("weights must be finite and non-negative, with a positive sum")
        # A first division by the largest weight keeps the sum finite: weights of 1e308 overflowed it. NumPy
        # divides exactly. XLA on the CPU multiplies by the reciprocal instead, and 1 / 1e308 is subnormal and
        # flushed to zero.
        w = w / w.max()
        weights = jnp.asarray(w / w.sum(), points.dtype)
    # An initial frame outside the span of the points joins the basis, so that the iteration stays exact.
    jb = joint_basis(points if init is None else jnp.concatenate([points, init[None]]))
    coords = jb.coordinates[:m]
    start = polar(jnp.einsum("m,mkp->kp", weights, coords)) if init is None else jb.coordinates[m]
    x, g_norm, k, ok = _frechet_loop(coords, weights, start, alpha, tol, max_iter, step, log_max_iter)
    return FrechetResult(lift(jb.basis, x), g_norm, k, ok)


@partial(jax.jit, static_argnames=("alpha", "tol", "max_iter"))
def _distances_to(centres, points, alpha, tol, max_iter):
    """``(M, k)`` matrix of distances from each point to each centre; NaN where a logarithm failed."""
    per_centre = jax.vmap(lambda c, y: distance(c, y, alpha=alpha, tol=tol, max_iter=max_iter), (None, 0))
    return jax.vmap(per_centre, (0, None), 1)(centres, points)


def pairwise_distance(
    points: Float[Array, "M n p"],
    *,
    alpha: float = 0.0,
    tol: float | None = None,
    max_iter: int = 100,
) -> Float[Array, "M M"]:
    """Symmetric matrix of geodesic distances with a zero diagonal.

    ``tol`` and ``max_iter`` are those of each logarithm (:func:`stiefel.log`).

    Ensures:
        THE CALLABLE SHALL compute each unordered pair once, in batches under ``vmap``, and SHALL put NaN
        where a logarithm did not converge (S2).
        IF a transformation differentiates through the callable THEN it SHALL raise ``TypeError`` (S5).
    """
    tol = _default_tol(tol, points.dtype)
    m = points.shape[0]
    out = jnp.zeros((m, m), points.dtype)
    if m < 2:
        return out
    coords = joint_basis(points).coordinates
    i, j = np.triu_indices(m, 1)
    # Each logarithm holds a few (K, 2p) arrays. Batches of about 2^22 entries bound the memory on a device.
    batch = max(1, min(len(i), 2**22 // (coords.shape[1] * coords.shape[2])))
    d = _pair_distances(coords, jnp.asarray(i), jnp.asarray(j), alpha, tol, max_iter, batch)
    return out.at[i, j].set(d).at[j, i].set(d)


@partial(jax.jit, static_argnames=("alpha", "tol", "max_iter", "batch"))
def _pair_distances(coords, i, j, alpha, tol, max_iter, batch):
    def one(ij):
        return distance(coords[ij[0]], coords[ij[1]], alpha=alpha, tol=tol, max_iter=max_iter)

    return jax.lax.map(one, jnp.stack([i, j], axis=1), batch_size=batch)


class KMeansResult(NamedTuple):
    labels: Int[Array, " M"]
    centers: Float[Array, "k n p"]
    #: Sum of squared geodesic distances of the points to their centres.
    inertia: Float[Array, ""]
    iterations: Int[Array, ""]
    #: True only if the labels stopped changing and every logarithm that was used had converged.
    converged: Bool[Array, ""]


def kmeans(
    key: PRNGKeyArray,
    points: Float[Array, "M n p"],
    n_clusters: int,
    *,
    alpha: float = 0.0,
    n_init: int = 1,
    max_iter: int = 50,
    tol: float | None = None,
    log_max_iter: int = 100,
) -> KMeansResult:
    """Riemannian k-means (Lloyd iteration with geodesic distances and Fréchet means as centres).

    ``tol`` is the tolerance of the Fréchet means that update the centres. When ``tol`` is None, it is
    ``1e-8`` in float64 and ``1e-3`` in float32. ``log_max_iter`` is the iteration limit of every logarithm,
    in the distances and in the means.

    Requires:
        THE CALLER SHALL pass ``1 <= n_clusters <= M``. IF this condition fails THEN THE CALLABLE SHALL
        raise ``ValueError``.
    Ensures:
        THE CALLABLE SHALL initialize the centres with the k-means++ rule on geodesic distances. It SHALL
        run ``n_init`` initializations and SHALL return the one with the smallest ``inertia``.
        THE CALLABLE SHALL return the same result for the same ``key`` (S7). It SHALL return labels in
        ``0 .. n_clusters - 1`` with every cluster non-empty, and centres that are frames (S1).
        IF a cluster becomes empty THEN THE CALLABLE SHALL re-seed its centre at one point of a cluster with
        more than one member. Among these points, it SHALL take the point farthest from its own centre.
        WHEN ``n_clusters == M`` THEN ``inertia`` SHALL be zero to rounding.
        THE CALLABLE SHALL return ``converged`` False also when a Fréchet mean of a centre did not converge.
        IF a transformation differentiates through the callable THEN it SHALL raise ``TypeError`` (S5).
    """
    m = points.shape[0]
    if not 1 <= n_clusters <= m:
        raise ValueError(f"n_clusters = {n_clusters} is not between 1 and M = {m}")
    tol = _default_tol(tol, points.dtype, 1e-8, 1e-3)
    jb = joint_basis(points)
    best = None
    for run_key in jax.random.split(key, n_init):
        run = _lloyd(run_key, jb.coordinates, n_clusters, alpha, max_iter, tol, log_max_iter)
        # A NaN inertia (from failed logarithms) is never chosen over a finite one.
        if best is None or bool(run.inertia < best.inertia) or bool(jnp.isnan(best.inertia)):
            best = run
    return best._replace(centers=lift(jb.basis, best.centers))


def _lloyd(key, coords, k, alpha, max_iter, tol, log_max_iter):
    m = coords.shape[0]
    log_tol = _default_tol(None, coords.dtype)

    def dist(centres):
        return _distances_to(centres, coords, alpha, log_tol, log_max_iter)

    # k-means++ (Arthur and Vassilvitskii 2007) on geodesic distances.
    key, sub = jax.random.split(key)
    chosen = [int(jax.random.randint(sub, (), 0, m))]
    d = dist(coords[chosen[0]][None])[:, 0]
    ok = bool(jnp.all(jnp.isfinite(d)))
    d2 = jnp.nan_to_num(d**2, nan=jnp.nanmax(d**2))
    for _ in range(1, k):
        key, sub = jax.random.split(key)
        total = jnp.sum(d2)
        probs = d2 / total if total > 0 else jnp.full(m, 1.0 / m)
        chosen.append(int(jax.random.choice(sub, m, p=probs)))
        new = dist(coords[chosen[-1]][None])[:, 0]
        ok &= bool(jnp.all(jnp.isfinite(new)))
        d2 = jnp.minimum(d2, jnp.nan_to_num(new**2, nan=jnp.nanmax(new**2)))
    centres = coords[jnp.asarray(chosen)]

    labels, stable, iterations = None, False, 0
    for step in range(1, max_iter + 1):
        iterations = step
        d = dist(centres)
        ok &= bool(jnp.all(jnp.isfinite(d)))
        d = np.asarray(jnp.where(jnp.isnan(d), jnp.inf, d))
        new_labels = d.argmin(axis=1)
        for empty in sorted(set(range(k)) - set(new_labels.tolist())):
            own = d[np.arange(m), new_labels]
            counts = np.bincount(new_labels, minlength=k)
            own[counts[new_labels] < 2] = -np.inf
            far = int(own.argmax())
            new_labels[far] = empty
            centres = centres.at[empty].set(coords[far])
        if labels is not None and np.array_equal(new_labels, labels):
            stable = True
            break
        labels = new_labels
        weights = jnp.asarray(labels[None, :] == np.arange(k)[:, None], coords.dtype)
        weights = weights / weights.sum(axis=1, keepdims=True)
        centres, _, _, mean_ok = jax.vmap(
            lambda w, c: _frechet_loop(coords, w, c, alpha, tol, 100, 1.0, log_max_iter)
        )(weights, centres)
        ok &= bool(jnp.all(mean_ok))
    d = dist(centres)
    ok &= bool(jnp.all(jnp.isfinite(d)))
    inertia = jnp.sum(d[np.arange(m), labels] ** 2)
    return KMeansResult(
        jnp.asarray(labels), centres, inertia, jnp.asarray(iterations), jnp.asarray(ok & stable)
    )
