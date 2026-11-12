"""Riemannian geometry and statistics on the Stiefel manifold St(n, p), written in JAX.

The functions are pure and work under ``jax.jit`` and ``jax.vmap``, except where their docstring states an
exception. The functions are meant for double precision: call ``jax.config.update("jax_enable_x64", True)``
before you create arrays.
"""

from stiefel.compress import JointBasis, joint_basis, lift
from stiefel.geometry import (
    LogResult,
    distance,
    exp,
    inner,
    is_frame,
    log,
    norm,
    polar,
    project_tangent,
    random_point,
    random_tangent,
    retract_polar,
    retract_qr,
)
from stiefel.manifold import Stiefel
from stiefel.statistics import (
    EMeansResult,
    FrechetResult,
    KMeansResult,
    emeans,
    frechet_mean,
    kmeans,
    pairwise_distance,
)

__version__ = "0.1.0"

__all__ = [
    "EMeansResult",
    "FrechetResult",
    "JointBasis",
    "KMeansResult",
    "LogResult",
    "Stiefel",
    "distance",
    "emeans",
    "exp",
    "frechet_mean",
    "inner",
    "is_frame",
    "joint_basis",
    "kmeans",
    "lift",
    "log",
    "norm",
    "pairwise_distance",
    "polar",
    "project_tangent",
    "random_point",
    "random_tangent",
    "retract_polar",
    "retract_qr",
]
