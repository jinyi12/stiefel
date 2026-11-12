"""Exact compression of frames to a joint subspace.

Let ``W`` be an ``(n, K)`` matrix with orthonormal columns. Then the map ``Y -> W Y`` embeds St(K, p) in
St(n, p). For every metric of the family in :mod:`stiefel.geometry`, this embedding is an isometry, and it
commutes with the exponential and with the logarithm:

    exp(W Y, W D) = W exp(Y, D),      log(W Y0, W Y1) = W log(Y0, Y1).

For frames in the range of ``W``, one can therefore compute geodesics, logarithms, distances, Fréchet means
and k-means on the small ``(K, p)`` frames. The cost of this computation does not depend on
``n``.
"""

import math
from functools import partial
from typing import NamedTuple

import jax
import jax.numpy as jnp
from jaxtyping import Array, Float

from stiefel.geometry import polar


class JointBasis(NamedTuple):
    #: Matrix of shape ``(n, K)``, with orthonormal columns; its range contains every input frame.
    basis: Float[Array, "n K"]
    #: Small frames, of shape ``(M, K, p)``, such that ``frames[i] = basis @ coordinates[i]`` holds up to
    #: the bound of :func:`joint_basis`.
    coordinates: Float[Array, "M K p"]


def joint_basis(frames: Float[Array, "M n p"], *, rtol: float | None = None) -> JointBasis:
    """Orthonormal basis of the span of all columns of ``frames``, and the coordinates of the frames in it.

    A direction is kept only if dropping it would move a column of a frame by more than a threshold in the
    2-norm. This threshold is ``500 * eps`` when ``rtol`` is None, and ``rtol`` otherwise. Since the columns
    of a frame have unit norm, ``rtol`` is relative to the column norm. To get the coordinates of a frame, the
    function takes the orthogonal polar factor of its projection ``basis^T frames[i]``. They are therefore
    frames even when a direction of the frame was dropped. The polar factor moves each column by at most
    ``p rtol^2`` more.

    There are two routes to the basis. In the fast route, one QR factorization of the stacked columns is
    taken, and then the SVD of its small triangular factor. Its result must pass a verification of the
    contract below: the basis must have finite entries, and the coordinates must be frames. Also, every column
    must be within the bound of the frame that it represents.

    If the result fails, Gram-Schmidt with reorthogonalization and column pivoting rebuilds the basis frame by
    frame (see :func:`_extend`). This second route uses matrix products only. Its result is verified in the
    same way. We need it because the LAPACK factorizations were observed to return NaN for many copies of one
    frame.

    Requires:
        THE CALLER SHALL pass ``M >= 1`` frames of one shape (S1), with finite entries, and a finite
        non-negative ``rtol``. IF one of these conditions fails THEN THE CALLABLE SHALL raise ``ValueError``.
    Ensures:
        THE CALLABLE SHALL return ``basis`` with orthonormal columns and ``coordinates`` that are frames (S1)
        on both routes. For every ``i``, ``max|basis @ coordinates[i] - frames[i]|`` SHALL be at most
        ``1e3 * eps`` when ``rtol`` is None (S3), and at most ``rtol (1 + p rtol) + 1e3 * eps`` otherwise.
        Every dropped part of a column is at most the threshold in the 2-norm, and the polar factor adds at
        most ``p rtol^2``. These bounds hold for exact duplicates, near-duplicates and generic frames.
        ``K`` SHALL satisfy ``min(n, 2p) <= K <= min(n, max(M, 2) p)``. WHEN the span has fewer than
        ``min(n, 2p)`` dimensions THEN THE CALLABLE SHALL complete the basis with orthonormal directions.
        WHEN ``M p >= n`` THEN THE CALLABLE SHALL return the identity basis and the frames themselves.
        IF the result of the Gram-Schmidt route fails the verification above THEN THE CALLABLE SHALL raise
        ``FloatingPointError`` instead of returning it. A basis that is not finite is an example of a failure.
        THE CALLABLE SHALL cost ``O(n (M p)^2)``. It SHALL NOT form an ``n x n`` array, and it SHALL
        factorize a matrix with ``n`` rows at most once (S6).
        The output shape depends on the data, so THE CALLABLE is not ``jit``-compatible (S5).
    """
    m, n, p = frames.shape
    if not bool(jnp.all(jnp.isfinite(frames))):
        raise ValueError("joint_basis: the frames have non-finite entries")
    if m * p >= n:
        return JointBasis(jnp.eye(n, dtype=frames.dtype), frames)
    threshold = 500 * float(jnp.finfo(frames.dtype).eps) if rtol is None else float(rtol)
    if not (math.isfinite(threshold) and threshold >= 0):
        raise ValueError(f"rtol = {rtol} must be finite and non-negative")
    largest = min(n, max(m, 2) * p)
    fast = _factorized_basis(frames, threshold)
    if fast is not None:
        return fast
    # The basis is kept in a zero-padded array whose width doubles when needed, so that the compiled step
    # can be used again. Zero columns do not change a projection, and a new width costs one compilation.
    basis, k = jnp.zeros((n, min(largest, 4 * p)), frames.dtype), jnp.asarray(0)
    for frame in frames:
        basis = _ensure_width(basis, int(k) + p, largest)
        basis, k = _extend(basis, k, frame, threshold, p)
    # Standard basis vectors complete the basis to min(n, 2p) directions. Among K + 2p of them, at least 2p
    # are independent of the K current directions, and the pivoting takes the least dependent ones.
    missing = min(n, 2 * p) - int(k)
    if missing > 0:
        candidates = jnp.eye(n, min(n, int(k) + 2 * p), dtype=frames.dtype)
        basis = _ensure_width(basis, int(k) + missing, largest)
        basis, k = _extend(basis, k, candidates, 0.0, missing)
    result = _verified(basis[:, : int(k)], frames, threshold)
    if result is None:
        raise FloatingPointError(
            "joint_basis: the Gram-Schmidt basis failed the verification of its contract"
        )
    return result


def _factorized_basis(frames, threshold):
    """The joint basis from one QR factorization and one small SVD, or None when it fails the verification.

    Let ``columns = Q R`` and ``R = L diag(s) V^T``. If the singular values at or below ``threshold`` are
    dropped, no column moves by more than ``threshold`` in the 2-norm. The choice of the rank therefore meets
    the contract of :func:`joint_basis`. Zero columns pad the matrix to ``2p`` columns. When the span is
    smaller, the Householder QR then also returns the directions that complete the basis. The cost is
    ``O(n (M p)^2)`` in matrix factorizations and products. This route is about ten times faster than the
    route that adds one column at a time.
    """
    m, n, p = frames.shape
    columns = jnp.moveaxis(frames, 0, 1).reshape(n, m * p)
    width = max(m * p, min(n, 2 * p))
    q, r = jnp.linalg.qr(jnp.pad(columns, ((0, 0), (0, width - m * p))))
    left, s, _ = jnp.linalg.svd(r)
    if not bool(jnp.all(jnp.isfinite(s))):
        return None
    rank = max(int(jnp.sum(s > threshold)), min(n, 2 * p))
    return _verified(q @ left[:, :rank], frames, threshold)


def _verified(basis, frames, threshold):
    """``basis`` and the polar factors of the projected frames as coordinates, or None when they fail the
    contract of :func:`joint_basis`.

    Let ``C = W^T X`` and ``R = X - W C``. Every column of ``R`` is at most ``t`` (the threshold) in the
    2-norm, so ``|R|_2 <= sqrt(p) t``. Because ``C^T C = I - R^T R``, the polar factor ``P`` of ``C``
    satisfies ``|P - C|_2 = 1 - sqrt(1 - |R|_2^2) <= p t^2``. Also, ``W (P - C)`` is orthogonal to ``R``, so
    that every column of ``W P - X`` is at most ``t (1 + p t)`` in the 2-norm. The test applies this bound
    to the result, with ``100 eps`` for rounding. It also checks that the coordinates are frames (S1), with
    a margin of two.
    """
    p = frames.shape[2]
    eps = float(jnp.finfo(frames.dtype).eps)
    # Without the polar factor a frame with a dropped direction had coordinates of squared column norm
    # 0.9975 for rtol = 0.1, which are not frames.
    coordinates = jax.vmap(polar)(jnp.einsum("nk,mnp->mkp", basis, frames))
    gram = jnp.einsum("mkp,mkq->mpq", coordinates, coordinates)
    defect = jnp.max(jnp.abs(gram - jnp.eye(p, dtype=frames.dtype)))
    error = jnp.linalg.norm(frames - jnp.einsum("nk,mkp->mnp", basis, coordinates), axis=1)
    ok = (
        jnp.all(jnp.isfinite(basis))
        & (defect <= 50 * p * eps)
        & (jnp.max(error) <= threshold * (1 + p * threshold) + 100 * eps)
    )
    return JointBasis(basis, coordinates) if bool(ok) else None


def _ensure_width(basis, width, largest):
    if width <= basis.shape[1]:
        return basis
    return jnp.pad(basis, ((0, 0), (0, min(largest, max(width, 2 * basis.shape[1])) - basis.shape[1])))


@partial(jax.jit, static_argnames=("limit",))
def _extend(basis, k, columns, threshold, limit):
    """``basis`` (zero beyond column ``k``) with at most ``limit`` new orthonormal columns. Every column of
    ``columns`` is within ``threshold`` (in the 2-norm) of the range of the result, up to rounding.

    The method is classical Gram-Schmidt with one reorthogonalization (Daniel, Gragg, Kaufman and Stewart
    1976, Math. Comp. 30) and column pivoting. The column with the largest residual joins the basis first.
    The method uses only matrix products. For 200 copies of one frame of St(4000, 5) with OpenBLAS threads,
    jnp.linalg.qr and both LAPACK SVD drivers returned NaN. The shapes that failed changed between runs, so
    the library factorizes no matrix with n rows here. The caller makes sure that ``basis`` has
    ``k + limit`` columns.
    """

    def project_out(x, w):
        # After two passes, the residual is orthogonal to w up to rounding, whatever the cancellation in the
        # first pass.
        for _ in range(2):
            x = x - w @ (w.T @ x)
        return x

    def pick(_, state):
        basis, k, residual = state
        norms = jnp.linalg.norm(residual, axis=0)
        j = jnp.argmax(norms)
        # Residuals only decrease. Once the largest one is below the threshold, every later step changes
        # nothing.
        accept = norms[j] > threshold
        q = project_out(residual[:, j] / jnp.where(accept, norms[j], 1.0), basis)
        q = jnp.where(accept, q / jnp.linalg.norm(q), 0.0)
        basis = jax.lax.dynamic_update_slice(basis, q[:, None], (0, k))
        return basis, k + accept, project_out(residual, q[:, None])

    state = (basis, k, project_out(columns, basis))
    basis, k, _ = jax.lax.fori_loop(0, min(limit, columns.shape[1]), pick, state)
    return basis, k


def lift(basis: Float[Array, "n K"], coordinates: Float[Array, "*batch K p"]) -> Float[Array, "*batch n p"]:
    """Frames ``basis @ coordinates`` in the ambient space, for any number of leading batch axes."""
    return jnp.einsum("nk,...kp->...np", basis, coordinates)
