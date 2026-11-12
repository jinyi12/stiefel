"""Fréchet means, pairwise distances and k-means for frames on the Stiefel manifold.

When a joint basis reduces the ambient dimension, iterations use the compressed coordinates (S3, S6).
The width of that basis depends on the data, so the functions cannot be compiled with ``jit`` as a whole.
Their inner loops are compiled.
"""

import math
from functools import partial
from typing import NamedTuple

import jax
import jax.numpy as jnp
import numpy as np
from jaxtyping import Array, Bool, Float, Int, PRNGKeyArray

from stiefel.compress import joint_basis, lift
from stiefel.geometry import _default_tol, _not_differentiable, distance, exp, is_frame, log, norm, polar


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

    Points with zero weight do not enter the mean, and failed logarithms for those points are ignored.
    Public mean and clustering calls omit zero-weight points before entering this loop.
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
        if not bool(jnp.all(jnp.isfinite(points))):
            raise ValueError("frechet_mean: the frames have non-finite entries")
        positive = w > 0
        points, w = points[positive], w[positive]
        m = len(points)
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
        for j in range(k):
            members = coords[np.flatnonzero(labels == j)]
            weights = jnp.full(len(members), 1 / len(members), coords.dtype)
            centre, _, _, mean_ok = _frechet_loop(
                members, weights, centres[j], alpha, tol, 100, 1.0, log_max_iter
            )
            centres = centres.at[j].set(centre)
            ok &= bool(mean_ok)
    d = dist(centres)
    ok &= bool(jnp.all(jnp.isfinite(d)))
    inertia = jnp.sum(d[np.arange(m), labels] ** 2)
    return KMeansResult(
        jnp.asarray(labels), centres, inertia, jnp.asarray(iterations), jnp.asarray(ok & stable)
    )


class EMeansResult(NamedTuple):
    labels: Int[Array, " M"]
    centers: Float[Array, "k n p"]
    inertia: Float[Array, ""]
    iterations: Int[Array, ""]
    #: True only after the final assignments and intrinsic mean gradients have been checked.
    converged: Bool[Array, ""]
    #: Number of point-centre logarithms evaluated, including refinement and final checks.
    log_evaluations: Int[Array, ""]


@partial(jax.jit, static_argnames=("alpha", "max_iter"))
def _nearest_centres(centres, points, alpha, max_iter):
    """Assignments and their logarithms; metric lower bounds avoid unnecessary candidate logarithms."""
    k, _, p = centres.shape
    eps = jnp.finfo(points.dtype).eps
    beta = 1 / (2 * (1 + alpha))
    margin = 100 * _default_tol(None, points.dtype) * math.sqrt(p) * max(1.0, math.sqrt(beta))

    def one(point):
        chord2 = jnp.sum((centres - point) ** 2, axis=(1, 2))
        # The metric dominates min(1, beta) times the ambient metric. Allow for roundoff in the bound.
        lower = jnp.sqrt(min(1.0, beta) * jnp.maximum(chord2 - 256 * p * eps, 0))
        order = jnp.argsort(lower)

        def cond(state):
            i, _, _, best, _, _ = state
            return (i < k) & (lower[order[jnp.minimum(i, k - 1)]] <= best * (1 + 32 * eps) + margin)

        def body(state):
            i, label, tangent, best, ok, calls = state
            j = order[i]

            def evaluate(_):
                result = log(centres[j], point, alpha=alpha, max_iter=max_iter)
                return result.tangent, norm(centres[j], result.tangent, alpha=alpha), result.converged, 1

            v, d, valid, count = jax.lax.cond(
                chord2[j] == 0,
                lambda _: (jnp.zeros_like(point), jnp.asarray(0, point.dtype), jnp.asarray(True), 0),
                evaluate,
                operand=None,
            )
            better = valid & (d < best)
            return (
                i + 1,
                jnp.where(better, j, label),
                jnp.where(better, v, tangent),
                jnp.where(better, d, best),
                ok & valid,
                calls + count,
            )

        initial = (0, jnp.asarray(0), jnp.zeros_like(point), jnp.asarray(jnp.inf, point.dtype), True, 0)
        _, label, tangent, d, ok, calls = jax.lax.while_loop(cond, body, initial)
        return label, d, tangent, ok & jnp.isfinite(d), calls

    # vmap would evaluate both sides of the conditional and defeat pruning.
    return jax.lax.map(one, points)


@jax.jit
def _ema_directions(labels, tangents, masses, step):
    """Count normalization of Bessa et al. (2026), applied to logarithms at the current centres."""
    k = masses.shape[0]
    counts = jnp.bincount(labels, length=k).astype(masses.dtype)
    sums = jax.ops.segment_sum(tangents, labels, k)
    updated = (1 - step) * masses + step * counts
    directions = sums * (step / jnp.where(updated > 0, updated, 1))[:, None, None]
    return directions, updated


@partial(jax.jit, static_argnames=("alpha", "max_iter"))
def _emeans_batch(centres, masses, points, step, alpha, max_iter):
    labels, d, tangents, valid, calls = _nearest_centres(centres, points, alpha, max_iter)
    directions, updated = _ema_directions(labels, tangents, masses, step)
    ok = jnp.all(valid)
    new_centres = jax.lax.cond(
        ok,
        lambda _: jax.vmap(lambda c, v: exp(c, v, alpha=alpha))(centres, directions),
        lambda _: centres,
        operand=None,
    )
    ok &= jnp.all(jnp.isfinite(new_centres)) & jnp.all(jax.vmap(is_frame)(new_centres))
    new_centres = jnp.where(ok, new_centres, centres)
    return new_centres, jnp.where(ok, updated, masses), d, ok, calls.sum()


def _chordal_seeds(key, points, k):
    """Arthur--Vassilvitskii (2007) seeding with ambient distances, without geodesic solves."""
    key, sub = jax.random.split(key)
    chosen = [int(jax.random.randint(sub, (), 0, len(points)))]
    squared = jnp.sum((points - points[chosen[0]]) ** 2, axis=(1, 2))
    for _ in range(1, k):
        key, sub = jax.random.split(key)
        available = jnp.ones(len(points)).at[jnp.asarray(chosen)].set(0)
        total = float(squared.sum())
        probabilities = squared / total if total > 0 else available / available.sum()
        chosen.append(int(jax.random.choice(sub, len(points), p=probabilities)))
        squared = jnp.minimum(squared, jnp.sum((points - points[chosen[-1]]) ** 2, axis=(1, 2)))
    return points[jnp.asarray(chosen)]


@jax.jit
def _projected_start(centres, points):
    """Five polar-mean steps for initialization; Bouchard et al. (2025), arXiv:2501.11555."""
    k = len(centres)

    def squared(c):
        return jnp.maximum(
            jnp.sum(points**2, axis=(1, 2))[:, None]
            + jnp.sum(c**2, axis=(1, 2))[None, :]
            - 2 * jnp.einsum("mnp,knp->mk", points, c),
            0,
        )

    def update(_, c):
        labels = jnp.argmin(squared(c), axis=1)
        counts = jnp.bincount(labels, length=k)
        sums = jax.ops.segment_sum(points, labels, k)
        means = jax.vmap(polar)(sums)
        return jnp.where((counts > 0)[:, None, None], means, c)

    centres = jax.lax.fori_loop(0, 5, update, centres)
    return centres, jnp.sum(jnp.min(squared(centres), axis=1))


def emeans(
    key: PRNGKeyArray,
    points: Float[Array, "M n p"],
    n_clusters: int,
    *,
    alpha: float = 0.0,
    epochs: int = 5,
    batch_size: int = 32,
    decay: float = 0.9,
    polish_iter: int = 5,
    tol: float | None = None,
    log_max_iter: int = 100,
) -> EMeansResult:
    """Intrinsic mini-batch clustering inspired by Bessa et al. (2026), arXiv:2611.09501.

    The objective is the sum of squared geodesic distances for the requested metric. Assignment logarithms
    are reused in mass-normalized exponential-map updates, with a cosine schedule on ``1 - decay``.
    Projected means initialize centres and ambient distances give bounds for pruning; neither replaces the
    geodesic objective. Each final mean uses only its assigned points and at most ``polish_iter`` steps.

    Requires:
        The caller must supply finite frames, ``1 <= n_clusters <= M``, positive integer ``epochs``,
        ``batch_size`` and ``log_max_iter``, non-negative integer ``polish_iter``, and ``0 <= decay < 1``.
        ``alpha`` must be finite and greater than -1. Invalid inputs raise ``ValueError``.
    Ensures:
        Centres are frames and every cluster is non-empty. The same key returns the same result.
        ``converged`` is True only if every evaluated logarithm succeeded, assignments have been checked
        against the final centres, and each assigned mean tangent has norm at most ``tol`` (default 1e-6
        in float64, 1e-3 in float32). Failure gives False and NaN inertia if an assignment was unreliable.
        Distances inherit the local-domain restriction of ``log``; global minimality is not guaranteed.
        On a curved manifold the update is not the exact exponentially weighted historical Frechet mean.
        Meta's Euclidean scaling laws and speedups are not assumed to transfer to this algorithm.
        Like the other statistics, this function cannot be differentiated or wrapped in ``jit`` as a whole.
    """
    points = _not_differentiable(jnp.asarray(points))
    if points.ndim != 3 or min(points.shape) < 1 or points.shape[1] < points.shape[2]:
        raise ValueError("points must have shape (M, n, p) with M >= 1 and n >= p >= 1")
    if not jnp.issubdtype(points.dtype, jnp.floating):
        raise ValueError("points must have a floating dtype")
    for name, value, minimum in (
        ("n_clusters", n_clusters, 1),
        ("epochs", epochs, 1),
        ("batch_size", batch_size, 1),
        ("polish_iter", polish_iter, 0),
        ("log_max_iter", log_max_iter, 1),
    ):
        if not isinstance(value, (int, np.integer)) or value < minimum:
            raise ValueError(f"{name} must be an integer >= {minimum}")
    if n_clusters > len(points):
        raise ValueError("n_clusters must not exceed the number of points")
    if not math.isfinite(alpha) or alpha <= -1:
        raise ValueError("alpha must be finite and greater than -1")
    if not math.isfinite(decay) or not 0 <= decay < 1:
        raise ValueError("decay must be finite and satisfy 0 <= decay < 1")
    if not bool(jnp.all(jnp.isfinite(points))) or not bool(jnp.all(jax.vmap(is_frame)(points))):
        raise ValueError("points must be finite frames with orthonormal columns")
    tol = _default_tol(tol, points.dtype, 1e-6, 1e-3)
    m, n, p = points.shape
    coords, basis = points, None
    if m * p < n:
        jb = joint_basis(points)
        coords, basis = jb.coordinates, jb.basis
    centres, best = None, math.inf
    key, seed_key = jax.random.split(key)
    for seed in jax.random.split(seed_key, 5):
        candidate, loss = _projected_start(_chordal_seeds(seed, coords, n_clusters), coords)
        if float(loss) < best:
            centres, best = candidate, float(loss)
    masses = jnp.full(n_clusters, 1e-10, points.dtype)
    batch_size = min(batch_size, m)
    updates = epochs * math.ceil(m / batch_size)
    calls, ok, update = 0, True, 0
    for _epoch in range(epochs):
        key, sub = jax.random.split(key)
        order = jax.random.permutation(sub, m)
        for start in range(0, m, batch_size):
            batch = coords[order[start : start + batch_size]]
            step = (1 - decay) * (0.1 + 0.45 * (1 + math.cos(math.pi * update / max(1, updates - 1))))
            centres, masses, distances, valid, evaluated = _emeans_batch(
                centres, masses, batch, step, alpha, log_max_iter
            )
            calls += int(evaluated)
            update += 1
            ok &= bool(valid)
            if not ok:
                break
            # Revive low-mass clusters at poorly represented batch points; all replacement centres are frames.
            if update % 5 == 0:
                for dead in np.flatnonzero(np.asarray(masses) < 0.1 * float(masses.mean())):
                    far = int(jnp.argmax(distances))
                    donor = int(jnp.argmax(masses))
                    mass = masses[donor] / 2
                    masses = masses.at[donor].set(mass).at[dead].set(mass)
                    centres = centres.at[dead].set(batch[far])
                    distances = distances.at[far].set(-jnp.inf)
        if not ok:
            break

    def assign():
        pieces = [
            _nearest_centres(centres, coords[i : i + batch_size], alpha, log_max_iter)
            for i in range(0, m, batch_size)
        ]
        return tuple(jnp.concatenate(x) for x in zip(*pieces, strict=True))

    labels, _, _, valid, evaluated = assign()
    calls += int(evaluated.sum())
    ok &= bool(jnp.all(valid))
    refined = True
    if ok and polish_iter:
        for j in range(n_clusters):
            members = coords[np.flatnonzero(np.asarray(labels) == j)]
            if len(members):
                weights = jnp.full(len(members), 1 / len(members), points.dtype)
                centre, _, steps, mean_ok = _frechet_loop(
                    members, weights, centres[j], alpha, tol, polish_iter, 1.0, log_max_iter
                )
                calls += len(members) * (int(steps) + 1)
                refined &= bool(mean_ok)
                if bool(jnp.all(jnp.isfinite(centre))) and bool(is_frame(centre)):
                    centres = centres.at[j].set(centre)
                else:
                    refined = False
    labels, distances, tangents, valid, evaluated = assign()
    calls += int(evaluated.sum())
    ok &= bool(jnp.all(valid))
    # As in Lloyd, resolve empty clusters, including ties between duplicate frames.
    moved = False
    for empty in sorted(set(range(n_clusters)) - set(np.asarray(labels).tolist())):
        counts = jnp.bincount(labels, length=n_clusters)
        far = int(jnp.argmax(jnp.where(counts[labels] > 1, distances, -jnp.inf)))
        moved |= bool(distances[far] > 0)
        labels = labels.at[far].set(empty)
        centres = centres.at[empty].set(coords[far])
        distances = distances.at[far].set(0)
        tangents = tangents.at[far].set(0)
    counts = jnp.bincount(labels, length=n_clusters)
    gradients = jax.ops.segment_sum(tangents, labels, n_clusters) / counts[:, None, None]
    stationary = bool(jnp.all(jax.vmap(lambda c, g: norm(c, g, alpha=alpha))(centres, gradients) <= tol))
    inertia = jnp.sum(distances**2) if ok else jnp.asarray(jnp.nan, points.dtype)
    return EMeansResult(
        labels,
        centres if basis is None else lift(basis, centres),
        inertia,
        jnp.asarray(_epoch + 1),
        jnp.asarray(ok and refined and stationary and not moved),
        jnp.asarray(calls),
    )
