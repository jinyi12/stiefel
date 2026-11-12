"""Object interface whose method names follow pymanopt, plus an adapter for pymanopt's optimizers.

``Stiefel(n, p)`` holds the shape and the metric parameter and forwards to the functions of
:mod:`stiefel.geometry`. A user of pymanopt or geomstats finds the operations under the names that are used
there:

==========================  ==================================  ===================================
this class                  pymanopt ``Stiefel``                geomstats ``Stiefel`` (canonical)
==========================  ==================================  ===================================
``projection(X, G)``        ``projection``                      ``to_tangent(G, X)``
``inner_product(X, A, B)``  ``inner_product`` (alpha = -1/2)    ``metric.inner_product(A, B, X)``
``exp(X, D)``               ``exp`` (alpha = -1/2)              ``metric.exp(D, X)``
``log(X, Y)``               not implemented there               ``metric.log(Y, X)``
``dist(X, Y)``              not implemented there               ``metric.dist(X, Y)``
``retraction(X, D)``        ``retraction`` (QR)                 ``metric.retraction(D, X)``
``random_point(key)``       ``random_point``                    ``random_uniform``
==========================  ==================================  ===================================
"""

import jax
import jax.numpy as jnp
import numpy as np
from jaxtyping import Array, Float, PRNGKeyArray

from stiefel import geometry
from stiefel.geometry import LogResult


class Stiefel:
    """The manifold St(n, p) with the metric of parameter ``alpha`` (0 canonical, -1/2 Euclidean)."""

    def __init__(self, n: int, p: int, *, alpha: float = 0.0):
        """
        Requires:
            THE CALLER SHALL pass ``n >= p >= 1`` and ``alpha > -1``. IF one of these conditions fails THEN
            THE CALLABLE SHALL raise ``ValueError``.
        """
        if not n >= p >= 1:
            raise ValueError(f"St(n, p) needs n >= p >= 1, got n = {n}, p = {p}")
        if not alpha > -1:
            raise ValueError(f"the metric needs alpha > -1, got {alpha}")
        self.n, self.p, self.alpha = n, p, float(alpha)

    @property
    def dim(self) -> int:
        """Dimension ``n p - p (p + 1) / 2`` of the manifold."""
        return self.n * self.p - self.p * (self.p + 1) // 2

    def projection(self, point: Float[Array, "n p"], vector: Float[Array, "n p"]) -> Float[Array, "n p"]:
        return geometry.project_tangent(point, vector)

    def inner_product(self, point, tangent_a, tangent_b) -> Float[Array, ""]:
        return geometry.inner(point, tangent_a, tangent_b, alpha=self.alpha)

    def norm(self, point, tangent) -> Float[Array, ""]:
        return geometry.norm(point, tangent, alpha=self.alpha)

    def exp(self, point, tangent) -> Float[Array, "n p"]:
        return geometry.exp(point, tangent, alpha=self.alpha)

    def log(self, point_a, point_b, **options) -> Float[Array, "n p"]:
        """Tangent vector at ``point_a``, filled with NaN when the logarithm did not converge (S2).

        Use :meth:`log_result` to obtain the residual and the convergence flag.
        """
        res = self.log_result(point_a, point_b, **options)
        return jnp.where(res.converged, res.tangent, jnp.nan)

    def log_result(self, point_a, point_b, **options) -> LogResult:
        return geometry.log(point_a, point_b, alpha=self.alpha, **options)

    def dist(self, point_a, point_b, **options) -> Float[Array, ""]:
        return geometry.distance(point_a, point_b, alpha=self.alpha, **options)

    def retraction(self, point, tangent) -> Float[Array, "n p"]:
        """QR retraction, as in pymanopt."""
        return geometry.retract_qr(point, tangent)

    def transport(self, point_a, point_b, tangent_a) -> Float[Array, "n p"]:
        """Vector transport by projection onto the tangent space at ``point_b``, as in pymanopt."""
        return geometry.project_tangent(point_b, tangent_a)

    def euclidean_to_riemannian_gradient(self, point, euclidean_gradient) -> Float[Array, "n p"]:
        """Riemannian gradient of the metric with parameter ``alpha`` from the Euclidean gradient.

        Ensures:
            THE CALLABLE SHALL return the tangent vector ``G`` with ``inner_product(point, G, D)`` equal to
            the Frobenius product of ``euclidean_gradient`` and ``D`` for every tangent ``D``.
        """
        # By eq. (5) of Zimmermann and Hüper (2022), the metric weighs the skew part U^T D by
        # 1 / (2 (alpha + 1)) and the normal part by 1. The skew part of the gradient is therefore rescaled.
        # The formula subtracts no two terms of similar size, so it stays accurate for any alpha > -1.
        skew = point.T @ euclidean_gradient
        skew = (skew - skew.T) / 2
        normal = euclidean_gradient - point @ (point.T @ euclidean_gradient)
        return 2 * (self.alpha + 1) * point @ skew + normal

    def random_point(self, key: PRNGKeyArray) -> Float[Array, "n p"]:
        return geometry.random_point(key, self.n, self.p)

    def random_tangent_vector(self, key: PRNGKeyArray, point) -> Float[Array, "n p"]:
        return geometry.random_tangent(key, point, alpha=self.alpha)

    def zero_vector(self, point) -> Float[Array, "n p"]:
        return jnp.zeros_like(point)

    def to_pymanopt(self, key: PRNGKeyArray | None = None):
        """A ``pymanopt.manifolds.manifold.Manifold`` that wraps this object (numpy arrays in and out).

        Since pymanopt draws random points without a key, the adapter splits ``key`` at every draw, so that a
        sequence of draws is reproducible (S7). When ``key`` is None, it starts from ``jax.random.key(0)``.

        Only first-order optimizers are supported, for example ``SteepestDescent`` and ``ConjugateGradient``.
        Because the adapter has no ``euclidean_to_riemannian_hessian``, ``TrustRegions`` and other
        second-order optimizers raise ``NotImplementedError`` in pymanopt.

        Ensures:
            THE CALLABLE SHALL import pymanopt only when it is called. IF pymanopt is missing THEN THE
            CALLABLE SHALL raise ``ImportError`` with an installation hint (``pip install stiefel[compat]``).
            THE CALLABLE SHALL leave ``jax_enable_x64`` as it found it (S4), although the import of pymanopt
            switches it on.
            In the tests, a pymanopt optimizer SHALL run on the returned manifold. One example is the problem
            of the dominant subspace, ``min -tr(X^T A X)``.
        """
        # When pymanopt is imported, its JAX backend is imported too, and this backend switches
        # jax_enable_x64 on for the whole process.
        x64 = jax.config.jax_enable_x64
        try:
            from pymanopt.manifolds.manifold import Manifold
        except ImportError as error:
            raise ImportError("Stiefel.to_pymanopt needs pymanopt: pip install stiefel[compat]") from error
        finally:
            jax.config.update("jax_enable_x64", x64)
        outer = self
        keys = [jax.random.key(0) if key is None else key]

        def draw():
            keys[0], sub = jax.random.split(keys[0])
            return sub

        def wrap(method):
            return lambda _self, *args: np.asarray(method(*(jnp.asarray(a) for a in args)))

        # The adapter must be a class, because pymanopt looks up the manifold operations as methods of a
        # Manifold subclass. Each method converts numpy arrays in and out, and it forwards to this object.
        class _PymanoptStiefel(Manifold):
            def __init__(self):
                super().__init__(f"Stiefel({outer.n}, {outer.p}), alpha = {outer.alpha}", outer.dim)

            @property
            def typical_dist(self):
                return float(np.sqrt(outer.p))

            inner_product = wrap(outer.inner_product)
            norm = wrap(outer.norm)
            projection = to_tangent_space = wrap(outer.projection)
            retraction = wrap(outer.retraction)
            exp = wrap(outer.exp)
            log = wrap(outer.log)
            dist = wrap(outer.dist)
            transport = wrap(outer.transport)
            euclidean_to_riemannian_gradient = wrap(outer.euclidean_to_riemannian_gradient)
            zero_vector = wrap(outer.zero_vector)

            def random_point(self):
                return np.asarray(outer.random_point(draw()))

            def random_tangent_vector(self, point):
                return np.asarray(outer.random_tangent_vector(draw(), jnp.asarray(point)))

        return _PymanoptStiefel()
