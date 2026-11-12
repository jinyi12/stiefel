"""Points, tangent vectors, geodesics and retractions on the Stiefel manifold St(n, p).

A point is an ``(n, p)`` array with orthonormal columns and is called a frame. A tangent vector at a frame
``U`` is an ``(n, p)`` array ``D`` for which ``U.T @ D`` is skew-symmetric.

The metric is the one-parameter family of Hüper, Markina and Silva Leite, in the parametrization of
Zimmermann and Hüper (2022, arXiv:2103.12046, eq. (5)):

    g_U(D, E) = tr(D^T (I - c U U^T) E),    c = (2 alpha + 1) / (2 (alpha + 1)),    alpha > -1.

``alpha = 0`` gives the canonical metric (the default, and the metric of geomstats'
``StiefelCanonicalMetric``). ``alpha = -1/2`` gives the Euclidean metric (the metric of pymanopt's
``Stiefel``). In every function, the order of the arguments is ``(base, tangent)`` or ``(base, point)``.

The global invariants S1 to S10 are stated in ``INVARIANTS.md``.

Sources of the algorithms (no third-party code is copied):

- Edelman, Arias and Smith (1998), SIAM J. Matrix Anal. Appl. 20(2), for the canonical exponential.
- Zimmermann (2017), SIAM J. Matrix Anal. Appl. 38(2), arXiv:1604.05054, for the algebraic logarithm.
- Zimmermann and Hüper (2022), SIAM J. Matrix Anal. Appl. 43(2), arXiv:2103.12046, for the metric family,
  its exponential, the Sylvester-accelerated logarithm and the shooting method.
- Mataigne, Zimmermann and Miolane (2024), arXiv:2403.11730, Appendix B, for the initial completion.
"""

import math
from functools import partial
from typing import NamedTuple

import jax
import jax.numpy as jnp
import jax.scipy.linalg as jsl
from jaxtyping import Array, Bool, Float, Int, PRNGKeyArray

Frame = Float[Array, "n p"]
Scalar = Float[Array, ""]


class LogResult(NamedTuple):
    """Outcome of :func:`log`. Every field is an array, so that the result passes through ``jit`` and
    ``vmap``."""

    tangent: Frame
    #: Endpoint residual, the Frobenius norm of ``exp(base, tangent) - point``, evaluated from the endpoint
    #: itself, so that it is right also for frames with a defect.
    residual: Scalar
    iterations: Int[Array, ""]
    #: True only if every entry is finite, ``tangent`` is tangent and ``residual`` is within its bound (see
    #: :func:`log`). The flag is a check of the result, and not only of the stopping test of the iteration.
    converged: Bool[Array, ""]


def _sym(a):
    return (a + a.T) / 2


def _skew(a):
    return (a - a.T) / 2


def _pad_rows(x, rows):
    """``x`` with zero rows appended up to ``rows``.

    This map is an isometric embedding of St(n, p) in St(rows, p). The exponential and the logarithm commute
    with it (see :mod:`stiefel.compress`).
    """
    return x if x.shape[0] >= rows else jnp.pad(x, ((0, rows - x.shape[0]), (0, 0)))


def _normal_basis(base, normal):
    """Orthonormal ``(n, p)`` columns that are orthogonal to ``base`` and whose range contains that of
    ``normal``, for ``n >= 2p``.

    A QR decomposition of ``normal`` alone is not sufficient when ``normal`` is rank deficient. Its Q factor
    then has columns that need not be orthogonal to ``base``. The Q factor of ``[base, normal]`` has
    orthonormal columns in every case. Its last ``p`` columns are orthogonal to the first ``p`` columns, whose
    range is that of ``base``. This is the only factorization of a matrix with ``n`` rows in a call (S6).
    """
    p = base.shape[1]
    return jnp.linalg.qr(jnp.concatenate([base, normal], axis=1))[0][:, p:]


def _exp_coords(a, b, alpha):
    """Coordinates ``[M; N]`` of ``Exp_U(U A + Q B) = U M + Q N``, Zimmermann and Hüper (2022), eq. (11).

    For ``alpha = 0`` the second factor is the identity and the formula is that of Edelman, Arias and Smith
    (1998), Corollary 2.2, eqs. (2.45)-(2.47).
    """
    p = a.shape[0]
    block = jnp.block([[a / (1 + alpha), -b.T], [b, jnp.zeros_like(a)]])
    return jsl.expm(block)[:, :p] @ jsl.expm(alpha / (1 + alpha) * a)


def _logm_orthogonal(v):
    """Real principal logarithm of an orthogonal matrix and a flag that is False when it is not reliable.

    Let ``V`` be orthogonal, with eigenvalues ``exp(i theta)``. The Cayley transform
    ``C = (V - I)(V + I)^{-1}`` is skew-symmetric, and ``H = i C`` is Hermitian with eigenvalues
    ``-tan(theta / 2)``. This map is injective on ``(-pi, pi)``. Hence ``log V = f(H)``, with
    ``f(x) = -2 i arctan(x)``, is a smooth function of a Hermitian matrix. Its evaluation with ``eigh`` is
    backward stable even for repeated eigenvalues. Also, ``eigh`` runs under ``jit`` on CPU and on CUDA, but
    a real Schur form does not.

    The flag is True only if the eigendecomposition reproduces ``V`` and no angle lies within about
    ``sqrt(eps)`` of pi. We need the second test because ``V + I`` is singular only up to rounding when -1
    is an exact eigenvalue. The transform then returns a huge but finite ``tan(theta / 2)`` that reproduces
    ``V`` well, although no real principal logarithm exists.
    """
    k = v.shape[0]
    eps = jnp.finfo(v.dtype).eps
    eye = jnp.eye(k, dtype=v.dtype)
    c = _skew(jnp.linalg.solve(v + eye, v - eye))
    lam, w = jnp.linalg.eigh(1j * c)
    log_v = _skew(jnp.real((w * (-2j * jnp.arctan(lam))) @ w.conj().T))
    rebuilt = jnp.real((w * ((1 - 1j * lam) / (1 + 1j * lam))) @ w.conj().T)
    error = jnp.max(jnp.abs(rebuilt - v))
    ok = jnp.isfinite(error) & (error <= 1e3 * k * eps) & (jnp.max(jnp.abs(lam)) <= 1 / jnp.sqrt(eps))
    return log_v, ok


def _completion(m, n_coords):
    """An orthogonal ``V = [[M, X0], [R^T N, P0]]`` in SO(2p) and the rotation ``R`` of the normal basis.

    The completion follows Mataigne, Zimmermann and Miolane (2024), Appendix B. It is rotated so that ``P0``
    is the diagonal of the singular values of the lower block of any completion. This is the choice closest
    to the identity, and it makes the initial ``C`` block small. If the determinant is -1, the column with
    the smallest singular value changes sign. The reason is that eq. (14) of Zimmermann and Hüper (2022)
    requires ``V`` in SO(2p).

    The completion is replaced by its orthogonal polar factor, which changes it only by rounding when the
    input frames are exact. Without this step, a frame defect of 3e-12 already failed the orthogonality test
    of :func:`_logm_orthogonal`. A defect of 3e-10 stopped the iteration at step 0.
    """
    p = m.shape[0]
    z = jnp.concatenate([m, n_coords])
    full = jnp.linalg.qr(z, mode="complete")[0]
    r, s, wt = jnp.linalg.svd(full[p:, p:])
    v = polar(jnp.block([[m, full[:p, p:] @ wt.T], [r.T @ n_coords, jnp.diag(s)]]))
    return v.at[:, -1].multiply(jnp.where(jnp.linalg.det(v) < 0, -1.0, 1.0).astype(v.dtype)), r


def _sym_sylvester(s, rhs, fallback):
    """Solution ``X`` of ``S X + X S = rhs`` for a symmetric ``S``, from one ``eigh`` of ``S``.

    In the eigenbasis of ``S``, the equation is diagonal with denominators ``s_i + s_j``. When one of them
    is closer to zero than 1/4, the solution is treated as unreliable, and ``fallback`` is returned instead.
    """
    w, e = jnp.linalg.eigh(s)
    den = w[:, None] + w[None, :]
    x = e @ ((e.T @ rhs @ e) / den) @ e.T
    return _skew(jnp.where(jnp.min(jnp.abs(den)) >= 0.25, x, fallback))


# The number of sub-iterations of Algorithm 4.2 of Mataigne, Zimmermann and Miolane (2024) per outer
# iteration. With the accelerated update (4.27), this is their pseudo-backward iteration (section 5.2).
# They report two sub-iterations as optimal for beta in [0.7, 1]. On St(12, 4), four sub-iterations keep
# alpha = 1 (beta = 1/4) within 100 outer iterations at distance 1. The accelerated update diverges for
# alpha = 1 at distances near pi / 2. There, twenty plain fixed-point updates (close to their backward
# iteration, Theorem 4.3) converge in 12 to 40 outer iterations for every pair that we drew. :func:`log`
# runs this second variant only where the first one failed.
_SUB_ITERATIONS = {True: 4, False: 20}


# The function is compiled once per shape and per static parameter. Otherwise, the closures of the loop
# would be traced and compiled again at every call outside jit.
@partial(jax.jit, static_argnames=("alpha", "tol", "max_iter", "accelerate"))
def _log_algebraic(m, n_coords, active, alpha, tol, max_iter, accelerate):
    """Mataigne, Zimmermann and Miolane (2024), Algorithm 4.1, with the first estimate (4.33).

    Each outer step refines ``A_hat`` by sub-iterations of their Algorithm 4.2. The sub-iterations use the
    accelerated update (4.27) when ``accelerate`` is True, and ``A_hat = A`` otherwise. The loop does not
    run where ``active`` is False.

    In their parametrization, ``2 beta = 1 / (alpha + 1)`` and ``tau = 1 - 2 beta = alpha / (alpha + 1)``.
    For ``alpha = 0``, the estimate ``A_hat`` drops out, and the loop is Algorithm 4 of Zimmermann and Hüper
    (2022) with the Sylvester step (19). The loop stops when ``|C| + |tau| |A_hat - A|`` is at most ``tol``.
    Both terms measure how far the factorization of ``V`` is from the geodesic form of eq. (12) of
    Zimmermann and Hüper (2022).
    """
    p = m.shape[0]
    two_beta, tau = 1 / (1 + alpha), alpha / (1 + alpha)
    eye = jnp.eye(p, dtype=m.dtype)
    v, rot = _completion(m, n_coords)

    def split(v, a_hat):
        # Eq. (4.2): the logarithm of V diag(expm(-tau A_hat), I).
        return _logm_orthogonal(v.at[:, :p].set(v[:, :p] @ jsl.expm(-tau * a_hat)))

    def gap(log_w, a_hat):
        return jnp.linalg.norm(log_w[p:, p:]) + abs(tau) * jnp.linalg.norm(a_hat - log_w[:p, :p] / two_beta)

    def cond(state):
        _, a_hat, log_w, ok, k = state
        return active & ok & (gap(log_w, a_hat) > tol) & (k < max_iter)

    def refine(log_w, a_hat):
        # Eq. (4.27): the accelerated forward estimate of A, or the plain fixed-point update.
        a = log_w[:p, :p] / two_beta
        return a - tau * jsl.expm(-tau * a) @ (a - a_hat) @ jsl.expm(tau * a) if accelerate else a

    def body(state):
        v, a_hat, log_w, _, k = state
        b, c = log_w[p:, :p], log_w[p:, p:]
        # Step 11: Gamma from the Sylvester equation (19) of Zimmermann and Hüper (2022). The fallback -C is
        # the step of Zimmermann (2017), Algorithm 1.
        gamma = _sym_sylvester(b @ b.T / 12 - eye / 2, c, -c)
        v = v.at[:, p:].set(v[:, p:] @ jsl.expm(gamma))
        a_new = refine(log_w, a_hat)
        log_new, ok = split(v, a_new)
        for _ in range(_SUB_ITERATIONS[accelerate] if tau != 0 else 0):
            a_new = refine(log_new, a_new)
            log_new, ok_sub = split(v, a_new)
            ok = ok & ok_sub
        # On failure the last finite iterate is kept, and the false flag ends the loop.
        return v, jnp.where(ok, a_new, a_hat), jnp.where(ok, log_new, log_w), ok, k + 1

    # Eq. (4.33): S A_hat + A_hat S = E with S = I / 2 - tau F^T F / 12, from log V0 = [[E, -F^T], [F, G]].
    log0, ok0 = _logm_orthogonal(v)
    f = log0[p:, :p]
    a_hat = _sym_sylvester(eye / 2 - tau * f.T @ f / 12, log0[:p, :p], log0[:p, :p])
    log_w, ok = split(v, a_hat)
    ok = ok0 & ok
    log_w = jnp.where(ok, log_w, 0.0)
    state = (v, a_hat, log_w, ok, jnp.asarray(0))
    _, a_hat, log_w, ok, k = jax.lax.while_loop(cond, body, state)
    return log_w[:p, :p] / two_beta, rot @ log_w[p:, :p], k, ok & (gap(log_w, a_hat) <= tol)


def _default_tol(tol, dtype, f64=1e-12, f32=1e-4):
    """``tol``, or the default of the data type when ``tol`` is None.

    A non-finite ``tol`` or a ``tol`` below ``100 eps`` raises ``ValueError`` (S4).
    """
    eps = float(jnp.finfo(dtype).eps)
    if tol is None:
        tol = f64 if dtype == jnp.float64 else f32
    tol = float(tol)
    # The test is written so that NaN fails it, because every comparison with NaN is False.
    if not (math.isfinite(tol) and tol >= 100 * eps):
        raise ValueError(
            f"tol = {tol} must be finite and at least 100 * eps = {100 * eps:.3g} of {jnp.dtype(dtype).name}"
        )
    return tol


@jax.custom_jvp
def _not_differentiable(x):
    """The identity, with a derivative rule that raises an error.

    The iterations of :func:`log` have no derivative that we can guarantee, and implicit differentiation is
    not implemented.
    """
    return x


@_not_differentiable.defjvp
def _not_differentiable_jvp(primals, tangents):
    raise TypeError(
        "stiefel.log is not differentiable, and neither are distance, pairwise_distance, frechet_mean, "
        "kmeans, emeans, Stiefel.log and Stiefel.dist, which call it: "
        "the logarithm is computed by an iteration. "
        "stiefel.exp is differentiable in forward and reverse mode."
    )


def is_frame(x: Frame, *, atol: float | None = None) -> Bool[Array, ""]:
    """Whether the columns of ``x`` are orthonormal.

    Ensures:
        THE CALLABLE SHALL return ``max|x^T x - I| <= atol``, with ``atol = 100 * p * eps(dtype)`` when
        ``atol`` is None.
    """
    p = x.shape[1]
    if atol is None:
        atol = 100 * p * float(jnp.finfo(x.dtype).eps)
    return jnp.max(jnp.abs(x.T @ x - jnp.eye(p, dtype=x.dtype))) <= atol


def project_tangent(base: Frame, vector: Frame) -> Frame:
    """Orthogonal projection (Frobenius inner product) of an ambient matrix onto the tangent space at
    ``base``.

    Ensures:
        THE CALLABLE SHALL return ``vector - base @ sym(base^T vector)``.
        WHEN ``vector`` is already tangent THEN THE CALLABLE SHALL return it unchanged up to rounding.
    """
    return vector - base @ _sym(base.T @ vector)


def inner(base: Frame, tangent_a: Frame, tangent_b: Frame, *, alpha: float = 0.0) -> Scalar:
    """Riemannian inner product ``g_base(tangent_a, tangent_b)`` of the metric with parameter ``alpha``.

    Requires:
        THE CALLER SHALL pass ``alpha > -1``.
    Ensures:
        THE CALLABLE SHALL evaluate the formula of the module docstring without forming an ``n x n`` array.
        It SHALL evaluate the formula as ``<A_a, A_b> / (2 (1 + alpha)) + <H_a, H_b>``, with
        ``A = base^T tangent`` and ``H = tangent - base A`` (Zimmermann and Hüper 2022, eq. (5)). This form
        has no cancellation for large ``alpha`` or for ``alpha`` near -1.
    """
    a_a, a_b = base.T @ tangent_a, base.T @ tangent_b
    # In the form tr(D^T E) - c tr(D^T U U^T E), the skew part is lost to cancellation when c is near 1.
    # At alpha = 1e16, that form returned 0 instead of 1e-16.
    normal_a, normal_b = tangent_a - base @ a_a, tangent_b - base @ a_b
    return jnp.sum(a_a * a_b) / (2 * (alpha + 1)) + jnp.sum(normal_a * normal_b)


def norm(base: Frame, tangent: Frame, *, alpha: float = 0.0) -> Scalar:
    """Riemannian norm ``sqrt(g_base(tangent, tangent))``, never NaN for a tangent input (clamped at zero)."""
    return jnp.sqrt(jnp.maximum(inner(base, tangent, tangent, alpha=alpha), 0.0))


def exp(base: Frame, tangent: Frame, *, alpha: float = 0.0) -> Frame:
    """Riemannian exponential ``Exp_base(tangent)`` of the metric with parameter ``alpha`` (closed form).

    The formula is that of Zimmermann and Hüper (2022), Proposition 1, eq. (11). For ``alpha = 0``, it is
    the formula of Edelman, Arias and Smith (1998), Corollary 2.2.

    Requires:
        THE CALLER SHALL pass a frame ``base`` (S1). IF ``alpha`` is a Python number that is not finite or
        is at most -1 THEN THE CALLABLE SHALL raise ``ValueError``. A traced ``alpha`` is not checked.
        A ``tangent`` that is not tangent at ``base`` is accepted. In that case, THE CALLABLE SHALL return
        ``exp(base, project_tangent(base, tangent))``: it drops the symmetric part of ``base^T tangent``
        without a warning.
    Ensures:
        Let ``A = skew(base^T tangent)`` and ``K = tangent - base base^T tangent``. The matrix exponentials
        act on ``[[A / (1 + alpha), -B^T], [B, 0]]`` with ``|B|_F = |K|_F``, and on ``alpha / (1 + alpha) A``.
        WHEN ``|A|_F / (1 + alpha) <= 1000``, ``|alpha| |A|_F / (1 + alpha) <= 1000`` and ``|K|_F <= 500``
        THEN THE CALLABLE SHALL return a frame (S1) for every ``n >= p >= 1``, including ``n < 2p`` and a
        rank-deficient normal component. In this domain, at most 0.84 of the S1 bound was measured. For
        ``-1/2 <= alpha <= 1``, the domain includes every tangent of Riemannian norm up to 500. For other
        ``alpha``, the Riemannian norm does not bound these quantities: ``alpha = 1e16`` and a tangent of norm
        1 give ``|alpha| |A|_F / (1 + alpha) = 1.4e8``.
        Beyond this domain, the frame error grows with the norms. An error of 1.03 times the S1 bound was
        measured at ``|alpha| |A|_F / (1 + alpha) = 2000``. The result is NaN once the 1-norm of an exponent
        reaches about ``7e5`` in float64 (``5e5`` in float32). This value is the limit of the 16 squarings
        of ``jax.scipy.linalg.expm``.
        WHEN ``tangent`` is zero THEN THE CALLABLE SHALL return ``base``.
        THE CALLABLE SHALL cost ``O(n p^2)`` and SHALL use matrix exponentials of size at most ``2p`` (S6).
        WHEN ``alpha == 0`` THEN the result SHALL agree with the formula of Edelman, Arias and Smith,
        ``[base Q] expm([[A, -R^T], [R, 0]]) [I; 0]``, with ``Q R`` the normal component.
        THE CALLABLE SHALL be differentiable in forward and reverse mode (``jax.jvp``, ``jax.grad``) at
        every tangent. This includes the zero tangent and tangents with a rank-deficient normal component.
        The derivative is that of the QR-free form
        ``[base, K / c] expm([[A / (1 + alpha), -K^T K / c], [c I, 0]]) [I; 0] expm(alpha / (1 + alpha) A)``,
        with ``c = max(|K|_F, 1)`` held constant. This form equals the exponential on frames for every
        ``c > 0``. The value is that of the QR form above. The derivative SHALL be finite in the domain of
        S1 stated above. It was checked against central differences up to ``|K|_F = 1000``.
    """
    # The test is written so that NaN fails it. A traced alpha (exp inside jit, with alpha as an argument)
    # cannot be checked here.
    if isinstance(alpha, int | float) and not (math.isfinite(alpha) and alpha > -1):
        raise ValueError(f"alpha = {alpha} must be finite and greater than -1")
    return _exp(base, tangent, alpha)


def _exp_qr(base, tangent, alpha):
    """Value of :func:`exp`, through an orthonormal basis of the normal component."""
    n, p = base.shape
    u, d = _pad_rows(base, 2 * p), _pad_rows(tangent, 2 * p)
    ud = u.T @ d
    normal = d - u @ ud
    q = _normal_basis(u, normal)
    coords = _exp_coords(_skew(ud), q.T @ normal, alpha)
    # The result lies in the span of base and tangent, so the zero rows that _pad_rows added stay zero up
    # to rounding.
    return (u @ coords[:p] + q @ coords[p:])[:n]


def _exp_smooth(base, tangent, alpha):
    """The same map without a QR decomposition, smooth in ``(base, tangent)``.

    Let ``K = Q B`` be the normal component, ``c > 0`` a scalar, ``S = diag(I, B / c)`` and
    ``M = [[A', -B^T], [B, 0]]``. Then ``M S = S M~`` for ``M~ = [[A', -B^T B / c], [c I, 0]]``. Hence
    ``[U Q] expm(M) [I; 0] = [U, K / c] expm(M~) [I; 0]``, and ``B^T B = K^T K``. This is eq. (11) of
    Zimmermann and Hüper (2022) after that similarity transformation. It does not give the primal value,
    because the block ``M~`` is not skew-symmetric and the factor ``[U, K / c]`` does not have orthonormal
    columns. For long tangents, these two properties cost accuracy.
    """
    p = base.shape[1]
    ud = base.T @ tangent
    a = _skew(ud)
    k = tangent - base @ ud
    # With c = 1, the block holds K^T K, whose norm grows as |K|^2. At |K| = 1000, the exponential of the
    # derivative overflowed its scaling range, and the JVP was NaN. With c = |K|, every entry is of the
    # order of |K|. The map does not depend on c, so the derivative stays exact when c is held fixed. The
    # floor of 1 keeps c positive at a zero normal component.
    c = jax.lax.stop_gradient(jnp.maximum(jnp.linalg.norm(k), 1.0))
    eye = jnp.eye(p, dtype=a.dtype)
    block = jnp.block([[a / (1 + alpha), -k.T @ k / c], [c * eye, jnp.zeros_like(a)]])
    left = jnp.concatenate([base, k / c], axis=1)
    return left @ jsl.expm(block)[:, :p] @ jsl.expm(alpha / (1 + alpha) * a)


# The QR factor of a rank-deficient normal component has no derivative (jax returned NaN at the zero
# tangent). The derivative therefore comes from the QR-free form, and the value comes from the accurate
# QR form.
_exp = jax.custom_jvp(_exp_qr)


@_exp.defjvp
def _exp_jvp(primals, tangents):
    # The primal output is computed by _exp again, so that a second differentiation also uses this rule.
    return _exp(*primals), jax.jvp(_exp_smooth, primals, tangents)[1]


def log(
    base: Frame,
    point: Frame,
    *,
    alpha: float = 0.0,
    tol: float | None = None,
    max_iter: int = 100,
) -> LogResult:
    """Riemannian logarithm: a tangent vector ``D`` at ``base`` with ``exp(base, D, alpha=alpha) = point``.

    For ``alpha == 0``, the function uses the algebraic algorithm of Zimmermann (2017) with the Sylvester
    acceleration of Zimmermann and Hüper (2022), Algorithm 4. For other ``alpha``, it uses the
    generalization by Mataigne, Zimmermann and Miolane (2024), Algorithm 4.1, which reduces to the first
    algorithm at ``alpha = 0``. Both algorithms start from the completion of Mataigne et al., Appendix B.
    ``alpha``, ``tol`` and ``max_iter`` must be Python numbers (static under ``jit``).

    The iteration is reliable for ``-1/2 <= alpha <= 0`` up to distances of about 2.5. For ``alpha = 1``, it
    converges up to distances of about 1, and mostly at 1.5. This value of ``alpha`` is ``beta = 1/4`` in
    Mataigne et al., the edge of their convergence theorem. For ``alpha > 1`` and for ``alpha`` close to -1,
    the iteration rarely converges beyond small distances. Each such failure gives ``converged`` False.

    Domain in ``alpha``. The iteration represents the geodesic by the principal logarithm of a ``2p x 2p``
    orthogonal matrix (Mataigne et al., eq. (4.2)). The upper left block of this matrix is
    ``A / (1 + alpha)``, with ``A = base^T tangent``. The iteration can therefore return only tangents for
    which every rotation angle of that matrix is below pi. Consider a point near the span of ``base`` that is
    reached by rotation angles ``theta_i``. For such a point, the condition is ``theta_i < pi (1 + alpha)``,
    which is ``pi / 2`` for ``alpha = -1/2``, although the geodesic exists and is unique up to ``pi``.

    Let a point lie exactly in the span, ``point = base @ R`` with ``R`` in SO(p) and no eigenvalue -1.
    Such a point is solved in closed form, ``tangent = base @ logm(R)``, because ``Exp_U(U A) = U expm(A)``
    for every ``alpha``. This case covers ``n = p``. Points with a normal component and rotation angles
    above ``pi (1 + alpha)`` are reported as not converged.

    Requires:
        THE CALLER SHALL pass ``base`` and ``point`` of the same shape, and ``alpha > -1``.
        THE CALLER SHALL pass frames whose defect ``max|x^T x - I|`` is at most
        ``max(min(tol, sqrt(eps)), 100 p eps)``. Every frame within S1 meets this bound. Because the
        completion is re-orthonormalized, such inputs give the accuracy of exact frames. IF a defect is larger
        THEN THE CALLABLE SHALL return ``converged`` False. The reason is that the endpoint test at that
        tolerance is meaningful only for frames that are orthonormal to within it.
        THE CALLER SHALL NOT expect the result to be the shortest geodesic when the points are farther apart
        than the injectivity radius. As Theorem 3.1 of Zimmermann (2017) shows, the algorithms converge
        locally.
    Ensures:
        WHEN ``tol`` is None THEN THE CALLABLE SHALL use ``1e-12`` in float64 and ``1e-4`` in float32.
        IF ``tol`` is not finite or ``tol < 100 * eps(dtype)`` (this includes zero, negative values and NaN)
        THEN THE CALLABLE SHALL raise ``ValueError`` (S4).
        ``converged`` SHALL be True if and only if all of these conditions hold:

        - every entry of ``tangent`` is finite;
        - ``tangent`` is tangent at ``base`` in the sense
          ``max|base^T tangent + tangent^T base| <= (1e3 eps + 2 p delta) max(1, max|tangent|)``, with
          ``delta`` the defect of ``base`` (so ``1e3 * eps`` relative accuracy for exact frames);
        - the input frames meet the defect bound above;
        - ``residual <= 100 * min(tol, sqrt(eps)) * sqrt(p)``.

        Whatever stopped the iteration (``tol`` met, ``max_iter`` reached, a non-finite value or an
        unreliable matrix logarithm), the status thus validates the returned tangent. When ``tol`` is above
        ``sqrt(eps)``, the iteration stops sooner, but the endpoint test is not loosened. For this reason, a
        loose ``tol`` cannot return ``converged`` True with a wrong endpoint.
        WHEN ``point`` equals ``base`` THEN THE CALLABLE SHALL return the zero tangent with ``converged``
        True, for every ``base`` within the defect bound above.
        ``residual`` SHALL be the Frobenius norm of the difference between ``point`` and the endpoint of the
        returned tangent. It SHALL be computed from that endpoint, at ``O(n p^2)`` cost. It is then right for
        every frame within the defect bound, and not only for orthonormal frames.
        WHEN ``converged`` is False THEN THE CALLABLE SHALL return the last finite iterate. It SHALL NOT
        raise, and it SHALL NOT report a NaN comparison as convergence (S2).
        ``iterations`` SHALL count the outer iterations of the accelerated variant. For ``alpha != 0``, it
        SHALL also count those of the fallback variant, which runs only when the first variant failed. Hence
        the count cannot exceed ``max_iter`` for ``alpha == 0``, or ``2 * max_iter`` otherwise. No iteration
        is added by the closed form for points in the span. Under ``jit`` and ``vmap``, however, the
        iteration is traced and run for every input, and ``iterations`` can then be non-zero for such points
        too.
        THE CALLABLE SHALL accept every ``n >= p >= 1``, including ``n < 2p`` and a rank-deficient normal
        component of ``point``.
        THE CALLABLE SHALL work under ``jax.jit`` and ``jax.vmap`` on CPU and on GPU (S5). It SHALL cost
        ``O(n p^2)`` plus ``O(p^3)`` per iteration, and it SHALL allocate no array larger than ``O(n p)``
        (S6).
        IF ``jax.jvp``, ``jax.grad`` or another transformation differentiates through the callable THEN
        it SHALL raise ``TypeError`` with the message that it is not differentiable (S5). We cannot guarantee
        a derivative of the iteration, and implicit differentiation is not implemented.
    """
    tol = _default_tol(tol, base.dtype)
    eps = float(jnp.finfo(base.dtype).eps)
    # A tolerance above sqrt(eps) makes the iteration stop sooner, but it does not loosen the endpoint
    # test. With tol = 1, a wrong endpoint therefore cannot be reported as converged.
    check_tol = min(tol, math.sqrt(eps))
    base, point = _not_differentiable(base), _not_differentiable(point)
    n, p = base.shape
    u, y = _pad_rows(base, 2 * p), _pad_rows(point, 2 * p)
    # point = U M + Q N up to rounding and the frame defect (Zimmermann and Hüper 2022, Theorem 3, eq. (13)).
    m = u.T @ y
    normal = y - u @ m
    q = _normal_basis(u, normal)
    n_coords = q.T @ normal
    a, b, iterations, ok = _log_algebraic(m, n_coords, True, alpha, tol, max_iter, True)
    if alpha != 0.0:
        # The fallback variant runs only where the accelerated variant failed. Under vmap, the other lanes are
        # idle.
        a2, b2, iterations2, ok2 = _log_algebraic(m, n_coords, ~ok, alpha, tol, max_iter, False)
        a, b = jnp.where(ok, a, a2), jnp.where(ok, b, b2)
        iterations = iterations + iterations2

    def endpoint_error(a, b):
        # The error of the endpoint itself, |U M + Q N - point|_F, at O(n p^2) cost and without a second
        # factorization of a matrix with n rows. A sum of squares of the coordinates would be exact only when
        # [U Q] is orthonormal. For a base with defect 9000 eps, that sum gave 2.8e-11 for log(U, U), whose
        # endpoint error is 0.
        coords = _exp_coords(a, b, alpha)
        return jnp.linalg.norm(u @ coords[:p] + q @ coords[p:] - y)

    bound = 100 * check_tol * math.sqrt(p)
    a = _skew(a)
    residual = endpoint_error(a, b)
    # For a point in the span of base, Exp(U A) = U expm(A) for every alpha, so U logm(U^T Y) is a logarithm.
    # The iteration cannot return it when an angle of A / (1 + alpha) passes pi, that is, for rotations
    # beyond pi / 2 at alpha = -1/2. This closed form therefore replaces the iterate when only the closed
    # form passes the test.
    a_span, ok_span = _logm_orthogonal(polar(m))
    zero = jnp.zeros_like(b)
    residual_span = endpoint_error(a_span, zero)
    use_span = ok_span & (residual_span <= bound) & ~(residual <= bound)
    a, b = jnp.where(use_span, a_span, a), jnp.where(use_span, zero, b)
    residual = jnp.where(use_span, residual_span, residual)
    tangent = (u @ a + q @ b)[:n]

    def defect(x):
        return jnp.max(jnp.abs(x.T @ x - jnp.eye(p, dtype=x.dtype)))

    # With base^T base = I + E, the symmetric part of base^T (base A) is (E A - A E) / 2, whose entries are at
    # most p |E| |A|. The tangency test allows twice that bound in base^T tangent + tangent^T base.
    base_defect = defect(base)
    skew_gap = base.T @ tangent
    scale = jnp.maximum(1.0, jnp.max(jnp.abs(tangent)))
    tangent_ok = jnp.max(jnp.abs(skew_gap + skew_gap.T)) <= (1e3 * eps + 2 * p * base_defect) * scale
    # The tolerance of the endpoint test is meaningful only when the frames are orthonormal to within it, so
    # inputs beyond this defect are rejected.
    frames_ok = jnp.maximum(base_defect, defect(point)) <= max(check_tol, 100 * p * eps)
    # The status validates the returned tangent, whatever stopped the iteration (a tolerance met, max_iter
    # reached, or an unreliable matrix logarithm). A NaN fails every comparison, so a NaN result is never
    # reported as converged.
    converged = jnp.all(jnp.isfinite(tangent)) & (residual <= bound) & tangent_ok & frames_ok
    return LogResult(tangent, residual, iterations, converged)


def distance(
    base: Frame,
    point: Frame,
    *,
    alpha: float = 0.0,
    tol: float | None = None,
    max_iter: int = 100,
) -> Scalar:
    """Length ``norm(base, log(base, point).tangent)`` of the geodesic that :func:`log` finds.

    Ensures:
        THE CALLABLE SHALL validate ``tol`` as :func:`log` does (``ValueError`` for a non-finite ``tol`` or
        one below ``100 * eps``).
        WHEN the logarithm has not converged THEN THE CALLABLE SHALL return NaN (S2).
        WHEN both logarithms converge THEN ``distance(a, b)`` and ``distance(b, a)`` SHALL agree to
        ``100 * tol``.
        IF a transformation differentiates through the callable THEN it SHALL raise ``TypeError``, as
        :func:`log` does (S5).
    """
    res = log(base, point, alpha=alpha, tol=tol, max_iter=max_iter)
    return jnp.where(res.converged, norm(base, res.tangent, alpha=alpha), jnp.nan)


def polar(matrix: Frame) -> Frame:
    """Nearest frame to a full-rank ``(n, p)`` matrix in the Frobenius norm: the factor ``U V^T`` of its
    SVD."""
    u, _, vt = jnp.linalg.svd(matrix, full_matrices=False)
    return u @ vt


def _qr_positive(x):
    q, r = jnp.linalg.qr(x)
    return q * jnp.where(jnp.diag(r) < 0, -1.0, 1.0).astype(x.dtype)


def retract_qr(base: Frame, tangent: Frame) -> Frame:
    """QR retraction: the Q factor of ``base + tangent`` with the diagonal of R made positive.

    Ensures:
        THE CALLABLE SHALL return a frame (S1) that agrees with pymanopt's ``Stiefel.retraction`` (S9).
    """
    return _qr_positive(base + tangent)


def retract_polar(base: Frame, tangent: Frame) -> Frame:
    """Polar retraction ``polar(base + tangent)``."""
    return polar(base + tangent)


def random_point(key: PRNGKeyArray, n: int, p: int, *, dtype=None) -> Frame:
    """A frame drawn from the uniform (Haar) distribution on St(n, p).

    Ensures:
        THE CALLABLE SHALL return the Q factor of an ``(n, p)`` standard normal matrix, with the diagonal of
        R made positive. This choice of signs makes the distribution uniform.
        THE CALLABLE SHALL return the same frame for the same ``key`` (S7).
    """
    return _qr_positive(jax.random.normal(key, (n, p), float if dtype is None else dtype))


def random_tangent(key: PRNGKeyArray, base: Frame, *, alpha: float = 0.0) -> Frame:
    """A tangent vector at ``base`` that is standard normal with respect to the metric with parameter
    ``alpha``.

    Ensures:
        THE CALLABLE SHALL return a tangent vector whose coordinates in a ``g``-orthonormal basis of the
        tangent space are independent standard normal variables. It SHALL do so without forming a complement
        of ``base``.
    """
    n, p = base.shape
    key_a, key_n = jax.random.split(key)
    # By eq. (5) of Zimmermann and Hüper (2022), g(U E, U E) = 1 / (alpha + 1) for E = e_i e_j^T - e_j e_i^T.
    # On the normal component, which is orthogonal to the skew part, the metric is the Frobenius product.
    g = jax.random.normal(key_a, (p, p), base.dtype)
    a = jnp.sqrt(1 + alpha) * (g - g.T) / jnp.sqrt(2.0)
    z = jax.random.normal(key_n, (n, p), base.dtype)
    return base @ a.astype(base.dtype) + (z - base @ (base.T @ z))
