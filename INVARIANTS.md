# Global invariants of `stiefel`

Each invariant spans several callables. The docstrings cite the invariants by number. Tolerances use `eps`, the
machine epsilon of the working data type.

## S1 — Frames stay frames
WHEN a public callable returns a point of the manifold THE SYSTEM SHALL keep `max|U^T U - I| <= 100 * p * eps`,
provided that its frame inputs satisfy the same bound.

For `exp`, this holds when the exponents of its matrix exponentials are bounded: `|A|_F / (1 + alpha) <= 1000`,
`|alpha| |A|_F / (1 + alpha) <= 1000` and `|K|_F <= 500`, with `A = skew(U^T D)` and `K = D - U U^T D`. For
`-1/2 <= alpha <= 1`, this domain includes every tangent of Riemannian norm up to 500. For other `alpha`, it
does not: at `alpha = 1e16`, a tangent of norm 1 can give NaN. `exp` raises `ValueError` for an `alpha` that is
not finite or is at most -1.

The coordinates of `joint_basis` are frames also when a user-given `rtol` drops a direction of a frame.
Participants: `exp`, `polar`, `retract_qr`, `retract_polar`, `random_point`, `joint_basis` (basis and coordinates),
`lift`, `frechet_mean`, `kmeans`, `Stiefel.exp`, `Stiefel.retraction`, `Stiefel.random_point`.

## S2 — No unvalidated logarithm
WHEN a logarithm has not converged THE SYSTEM SHALL make that visible to the caller. `converged` is True only for
a validated result (finite, tangent, endpoint residual within its bound), whatever stopped the iteration. For a
logarithm that has not converged, `log` returns `converged = False`, and `distance` and `pairwise_distance`
return NaN. `Stiefel.log` returns NaN, and `frechet_mean` and `kmeans` return `converged = False`. No callable
prints a warning in place of a status, and no callable compares a NaN and reads the result as convergence.
Participants: `log`, `distance`, `pairwise_distance`, `frechet_mean`, `kmeans`, `Stiefel.log`, `Stiefel.dist`.

## S3 — Compression is exact
WHEN frames lie in the range of a joint basis `W` THE SYSTEM SHALL give the same results on the small and the
ambient frames. The agreement is to `1e-10` in float64. It applies to exponentials, logarithms,
distances, means and clusters.
Participants: `joint_basis`, `lift`, `exp`, `log`, `distance`, `frechet_mean`, `pairwise_distance`, `kmeans`.

## S4 — Precision
THE SYSTEM SHALL compute in the data type of its inputs and SHALL NOT change JAX's global configuration. This
includes `Stiefel.to_pymanopt`, whose import of pymanopt switches on `jax_enable_x64`. A tolerance that is not
finite or is below `100 * eps` of that data type raises `ValueError`. Every tolerance has a default for float64
and one for float32. The tests run in float64, except one test that runs the public callables with JAX's
defaults.
Participants: every callable.

## S5 — Transformations and devices
THE SYSTEM SHALL keep the callables of `stiefel.geometry` compatible with `jax.jit` and `jax.vmap`, on CPU and on
a CUDA device. Some callables have an output shape that depends on the data: `joint_basis`, and through it the
callables of `stiefel.statistics`. Their docstrings state this. No callable calls back into NumPy or SciPy inside
a traced function.

`exp` SHALL be differentiable in forward and reverse mode at every tangent. This includes the zero tangent and a
rank-deficient normal component. The derivative SHALL be finite in the domain of S1 for `exp`.

`log`, `distance`, `pairwise_distance`, `frechet_mean`, `kmeans`, `Stiefel.log` and `Stiefel.dist` iterate. WHEN
a transformation differentiates through them, in forward or in reverse mode, they SHALL raise `TypeError` with a
message that gives this reason. Implicit differentiation is not implemented.
Participants: every callable of `stiefel.geometry`; for differentiation, also the callables named above.

## S6 — Cost
THE SYSTEM SHALL NOT allocate an `n x n` array and SHALL NOT factorize a matrix with `n` rows more than once
per call. `joint_basis` verifies the result of its factorization. When the verification fails, it falls back to
Gram-Schmidt with matrix products. The result of Gram-Schmidt is verified in the same way, and a failure raises
an error. Matrix exponentials, logarithms and Sylvester solves act on matrices of size at most `2p`.
Participants: every callable.

## S7 — Explicit randomness
THE SYSTEM SHALL take a JAX key wherever it draws random numbers and SHALL return the same output for the same
key.
Participants: `random_point`, `random_tangent`, `kmeans`, `Stiefel.random_point`, `Stiefel.random_tangent_vector`.

## S8 — Library hygiene
THE SYSTEM SHALL NOT print, write files or read environment variables.
Participants: every callable.

## S9 — Agreement with other packages
WHEN `alpha = 0` THE SYSTEM SHALL agree with geomstats' canonical metric for `exp`, `log` and the inner product,
to `1e-8`. WHEN `alpha = -1/2` THE SYSTEM SHALL agree with pymanopt's `Stiefel` for `exp`, `projection`,
`inner_product` and `retraction`, to `1e-8`. The tests that check this are skipped when the other package is not
installed.
Participants: `exp`, `log`, `inner`, `project_tangent`, `retract_qr`, `Stiefel`.

## S10 — Own code
THE SYSTEM SHALL contain no code copied from another project. For each algorithm, the docstring of the callable
that implements it names the published source.
Participants: every module.
