# Numerical notes

## The metric family


For a frame `U` and tangent vectors `D` and `E` at `U` (that is, `U^T D` and `U^T E` are skew-symmetric), the
metric with parameter `alpha > -1` is

```
g_U(D, E) = tr(D^T (I - c U U^T) E),    c = (2 alpha + 1) / (2 (alpha + 1))
```

Zimmermann and Hüper (2022) give it as their eq. (5). Every function takes the keyword argument `alpha`, and
its value is 0 when it is omitted.

| `alpha` | `c` | metric | same as |
|---:|---:|---|---|
| `0` | `1/2` | canonical | geomstats `StiefelCanonicalMetric` |
| `-1/2` | `0` | Euclidean (Frobenius) | pymanopt `Stiefel` |

In geomstats, the canonical inner product is exactly `tr(D^T (I - U U^T / 2) E)`, with no extra factor.
Hence `stiefel.inner` at `alpha = 0` returns the same number.

## Functions

In the table, a frame has size `n x p`, and `M` is the number of frames. `K` is the width of the joint
basis, which is at most `M p`.

| function | computes | cost |
|---|---|---|
| `random_point(key, n, p)` | uniform (Haar) frame | `O(n p^2)` |
| `random_tangent(key, base, alpha=)` | tangent vector that is standard normal for the metric | `O(n p^2)` |
| `is_frame(x)` | whether `max abs(x^T x - I) <= 100 p eps` | `O(n p^2)` |
| `project_tangent(base, v)` | orthogonal projection onto the tangent space | `O(n p^2)` |
| `inner(base, a, b, alpha=)`, `norm` | Riemannian inner product and norm, from the split of each tangent into `base A + H` | `O(n p^2)` |
| `exp(base, tangent, alpha=)` | Riemannian exponential, in closed form; a non-tangent input is first projected onto the tangent space | `O(n p^2)`, matrix exponentials of size `2p` |
| `log(base, point, alpha=, tol=, max_iter=)` | Riemannian logarithm and its status (`LogResult`) | `O(n p^2)` plus `O(p^3)` per iteration |
| `distance(base, point, alpha=)` | length of the geodesic that `log` finds; NaN on failure | as `log` |
| `retract_qr`, `retract_polar`, `polar` | QR and polar retractions, nearest frame | `O(n p^2)` |
| `joint_basis(frames, rtol=)` | orthonormal basis `W` of the span of all frames, and frames `Y_i` with `X_i = W Y_i` | `O(n K M p)`, at most `O(n (M p)^2)` |
| `lift(basis, coordinates)` | `basis @ coordinates` over any batch axes | `O(n K p)` per frame |
| `pairwise_distance(points, alpha=)` | symmetric matrix of distances; NaN where a logarithm failed | `joint_basis` plus `M (M - 1) / 2` logarithms on St(K, p) |
| `frechet_mean(points, weights=, init=, alpha=, tol=, log_max_iter=)` | stationary point of the weighted sum of squared distances (`FrechetResult`) | `joint_basis` plus `M` logarithms on St(K, p) per iteration |
| `kmeans(key, points, k, alpha=, n_init=, tol=, log_max_iter=)` | Riemannian k-means (`KMeansResult`) | `joint_basis` plus `M k` distances and `k` Fréchet means per step |
| `emeans(key, points, k, alpha=, epochs=, batch_size=, decay=, polish_iter=)` | intrinsic mini-batch clustering (`EMeansResult`) | pruned assignments, one update per batch and final mean refinement |
| `Stiefel(n, p, alpha=)` | object with pymanopt's method names, and `to_pymanopt()` | as the functions |

When `M p < n`, joint compression avoids an `n x n` array and factorizes a matrix with `n` rows
at most once (invariant S6). Otherwise `joint_basis` returns an explicit identity basis and the original
frames. In that case `emeans` uses the frames directly, without requesting a joint basis.

To build its basis, `joint_basis` takes one QR factorization of the stacked columns and then the SVD of the
small triangular factor. It returns that basis only after it has verified the contract on it. The basis must
have finite entries, and the coordinates must be frames. Also, every column must be within the bound of the
frame that it represents.

If the verification fails, `joint_basis` rebuilds the basis by Gram–Schmidt with column pivoting and one
reorthogonalization. This route uses only matrix products. Its result is verified in the same way, and
`FloatingPointError` is raised if it fails. We added the second route after a failure that we observed. For
200 copies of one frame of St(4000, 5) with OpenBLAS threads, the LAPACK QR and both SVD drivers returned NaN.

A direction is kept only if dropping it would move a column of a frame by more than `500 * eps` in the
2-norm. When the caller gives `rtol`, the threshold is `rtol` instead. Since the coordinates of a frame are
the polar factor of its projection onto the basis, they are frames even when a direction was dropped. Each
column of `basis @ coordinates[i]` is then within `rtol (1 + p rtol)` of the frame, plus rounding.

## Exact compression

**Statement.** Let `W` be an `n x K` matrix with orthonormal columns. For every metric of the family,

```
exp(W Y, W D) = W exp(Y, D),        log(W Y0, W Y1) = W log(Y0, Y1),
```

and the distance between `W Y0` and `W Y1` on St(n, p) equals the distance between `Y0` and `Y1` on St(K, p).

**Why it holds.** Because `W^T W = I`, the metric at `W Y` on the tangent vectors `W D` and `W E` reduces to
the same expression in `Y`, `D` and `E`. The map from `Y` to `W Y` is therefore an isometric embedding. For a
frame and a tangent vector in the range of `W`, the closed-form exponential stays in that range. Hence the
geodesics of the small frames are those of the large frames, and so are their logarithms.

```python
import time
import jax

jax.config.update("jax_enable_x64", True)
import stiefel as sf

n, p, M = 100_000, 4, 64
frames = jax.vmap(lambda k: sf.random_point(k, n, p))(jax.random.split(jax.random.key(1), M))
t0 = time.perf_counter()
jb = sf.joint_basis(frames)  # basis (100000, 256), coordinates (64, 256, 4)
small = sf.log(jb.coordinates[0], jb.coordinates[1]).tangent  # computed on St(256, 4)
print(abs(sf.lift(jb.basis, small) - sf.log(frames[0], frames[1]).tangent).max())
D = sf.pairwise_distance(frames)  # 2016 logarithms, all on St(256, 4)
print(jb.basis.shape, f"{time.perf_counter() - t0:.1f} s")
```

We ran this code with jax 0.12.0.1 in float64 on an AMD Ryzen 9 7950X, limited to 8 threads. Other jobs
ran on the same machine at that time. The two logarithms differed by `8.7e-17`. In each of two runs, the timed part
took 4.9 s, and this time includes compilation.

When it was timed alone, one call of `joint_basis` took 1.0 to 1.3 s. If the fast route fails its
verification and the Gram–Schmidt route must run instead, the same call takes about 10 s. We also timed one
compiled logarithm by itself. Over 5 calls, its median was 6.7 ms on St(100000, 4) and 0.92 ms on St(256, 4).

**Consequence for statistics.** Suppose that we have `M` frames of size `n x p`. We pay for the dependence
on `n` once, in the cost of `joint_basis`: `O(n K M p)`, which is at most `O(n (M p)^2)`. Every later
logarithm, distance, mean or k-means step then costs what it would cost on St(K, p), with `K <= M p`. Before
they iterate, `frechet_mean`, `pairwise_distance` and `kmeans` compress their input in this way. The tests
check (invariant S3) that their results agree to `1e-10` with the results on the ambient frames. The gain is
large if `M p` is much smaller than `n`, as it is for the reduced bases of a parametric reduced-order model.

## Fréchet mean and k-means

`frechet_mean` iterates `X <- exp(X, step * sum_i w_i log(X, points[i]))`, starting from the polar factor of
the weighted arithmetic mean. It returns `converged = True` only if every logarithm that it used has converged.
The Riemannian norm of the mean tangent vector must also be at most `tol`.

By default, `tol` is `1e-10` in float64 and `1e-3` in float32. Each logarithm runs at most
`log_max_iter = 100` iterations, but frames that are far apart can need more. If the weights are not finite,
are negative or sum to zero, `ValueError` is raised.

- Guaranteed:
  - the result is a frame;
  - `converged` is False whenever a logarithm failed or the iteration ran out of steps;
  - the compression of the input does not change the result beyond `1e-10`.
- Not guaranteed: that the stationary point is the global minimizer of the sum of squared distances, or that
  the iteration converges at all. The iteration is local. The mean is unique when the points lie in a
  geodesic ball whose radius is below the convexity radius (Afsari 2011). The library does not check this
  condition.

`kmeans` runs the Lloyd iteration with geodesic distances, and it takes Fréchet means as centres. Its start
is a k-means++ initialization on geodesic distances. This is done `n_init` times, and the run with the
smallest inertia is kept. If a cluster is empty, it is re-seeded at the farthest point of a cluster that has
more than one member. Its `tol` is passed to the Fréchet means of the centres; by default it is `1e-8` in
float64 and `1e-3` in float32. Every logarithm in the distances and the means is bounded by `log_max_iter`.

- Guaranteed:
  - the same result for the same key;
  - labels in `0 .. k - 1`, with no empty cluster;
  - centres that are frames;
  - `converged = False` if a logarithm or the Fréchet mean of a centre failed, or if the labels did not
    settle within `max_iter` steps.
- Not guaranteed: the global minimum of the inertia, which k-means does not reach in general even in the
  Euclidean case.

Neither function can be wrapped in `jit` as a whole, because the width `K` of the joint basis depends on the
data. Their inner loops are compiled.

The [fast-clustering note](fast-clustering.md) describes the Riemannian e-means adaptation and its final checks.

## The logarithm: domain and failure reports

`log(base, point)` returns a `LogResult` with four arrays:

- `tangent`: the last iterate, a tangent vector at `base`.
- `residual`: the Frobenius norm of `exp(base, tangent) - point`, evaluated from the endpoint itself at
  `O(n p^2)` cost. It is therefore right also for input frames with a defect within the bound of the
  docstring.
- `iterations`: the number of outer iterations. For `alpha != 0`, it is the sum over the accelerated variant
  and, where that variant failed, the plain fixed-point variant. It can therefore reach `2 * max_iter`.
- `converged`: True if and only if the returned result is validated. A validated result meets all of these
  conditions:
  - every entry of `tangent` is finite;
  - `tangent` is tangent at `base` to `1e3 * eps` relative accuracy;
  - the defects `max abs(x^T x - I)` of both input frames are within the bound of the docstring;
  - `residual <= 100 * min(tol, sqrt(eps)) * sqrt(p)`.

The status is computed in the same way, whatever stopped the iteration: `tol` met, `max_iter` reached, a
non-finite value, or an unreliable matrix logarithm. A `tol` above `sqrt(eps)` stops the iteration sooner but
does not loosen the endpoint test. The default `tol` is `1e-12` in float64 and `1e-4` in float32. A `tol` that
is not finite, not positive or below `100 * eps` of the data type raises `ValueError`. This applies to `log`
and to every function that takes a tolerance.

A True `converged` certifies that the tangent vector reaches `point`. It does not certify that the geodesic is
the shortest one, which holds only within the injectivity radius.

**Points in the span of the base.** Let `point = base @ R`, with `R` a rotation without eigenvalue -1. If the
iteration fails, `log` falls back on the closed form `base @ logm(R)`, because `exp(U, U A) = U expm(A)` for
every `alpha`. We checked this on St(12, 4), with angles 2.0, 3.0 and 3.1 in one plane. For these rotation
angles up to pi, the result converged for `alpha` in {-1/2, 0, 1, 2}.

It failed for `alpha = -0.8` at angle 2.0, and we report this to the maintainers. The endpoint test evaluates
`expm` of a skew-symmetric matrix of norm `2.0 / (1 + alpha) = 10`. For this matrix, `jax.scipy.linalg.expm` is
accurate only to about `3e-9`, which is above the bound `2e-10`. For `alpha != 0`, the iteration runs first, so
`iterations` reaches 200 before the closed form is used.

**No unvalidated logarithm (invariant S2).** When a logarithm has not converged, the functions return the
following:

- `log` returns `converged = False`;
- `distance` and `pairwise_distance` return NaN;
- `Stiefel.log` returns an array of NaN;
- `frechet_mean` and `kmeans` return `converged = False`.

No function prints a warning in place of a status, and none raises on non-convergence.

**Documented condition.** The iteration represents the geodesic by the principal logarithm of a `2p x 2p`
orthogonal matrix (Mataigne et al., eq. (4.2)). The upper left block of this matrix is `A / (1 + alpha)`, with
`A = base^T tangent`. For a point with a normal component, the iteration can therefore succeed only if every
rotation angle `theta_i` of that matrix is below pi. Near the span of `base`, this means
`theta_i < pi (1 + alpha)`, which is pi / 2 for `alpha = -1/2`. Points that violate this condition are reported
as not converged.

**Measured domain.** The tests check the round trip `log(base, exp(base, D)) = D` to `1e-10`. They use the
distances 0.1, 1.0 and 1.5, `alpha` in {-1/2, 0, 1}, and the shapes St(12, 4), St(5, 3) and St(6, 1). We also
ran a wider sweep on the current code, with float64, jax 0.12.0.1 and `max_iter = 100`.

For each cell of the table, we drew 20 pairs. Each pair is a uniform frame and the endpoint of a random
tangent vector, rescaled to the distance of the column. The tangent vector is standard normal for the metric,
as drawn by `random_tangent`. For such tangents, the skew block `A` is small compared with the normal block,
so the condition above rarely limits the result. Tangents that are mostly rotations inside the span of `base`
reach that limit at smaller distances.

The table gives the number of pairs for which `converged` was True. In parentheses, it gives the number of
pairs for which the result was also the tangent vector that generated the endpoint, to `1e-8`.

| `alpha` | shape | 0.5 | 1.0 | 1.5 | 2.0 | 2.5 | 3.0 |
|---:|---|---:|---:|---:|---:|---:|---:|
| -1/2 | St(12, 4) | 20 (20) | 20 (20) | 20 (20) | 20 (20) | 20 (20) | 18 (18) |
| 0 | St(12, 4) | 20 (20) | 20 (20) | 20 (20) | 20 (20) | 20 (20) | 19 (19) |
| 1 | St(12, 4) | 20 (20) | 20 (20) | 20 (20) | 20 (20) | 20 (20) | 18 (8) |
| -0.8 | St(12, 4) | 20 (20) | 20 (20) | 20 (20) | 20 (20) | 11 (11) | 4 (4) |
| 2 | St(12, 4) | 18 (18) | 5 (5) | 0 | 0 | 0 | 0 |
| -1/2 | St(30, 5) | 20 (20) | 20 (20) | 20 (20) | 20 (20) | 20 (20) | 20 (20) |
| 0 | St(30, 5) | 20 (20) | 20 (20) | 20 (20) | 20 (20) | 20 (20) | 20 (20) |
| 1 | St(30, 5) | 20 (20) | 20 (20) | 20 (20) | 20 (20) | 20 (20) | 20 (20) |
| -0.8 | St(30, 5) | 20 (20) | 20 (20) | 20 (20) | 20 (20) | 20 (20) | 17 (17) |
| 2 | St(30, 5) | 20 (20) | 10 (10) | 2 (2) | 1 (1) | 0 | 0 |

The counts are the same as for the earlier version of the library, with one exception. For `alpha = 0` on
St(12, 4) at distance 3, one more pair now converges. Up to distance 2.5, the round trip succeeded for every sampled
pair when `alpha` was in [-1/2, 1]. At distance 3 with `alpha = 1` on St(12, 4), 10 of the 18 converged
results were another geodesic to the same endpoint.

Outside [-1/2, 1], the domain is smaller. At `alpha = 2`, the iteration rarely converged beyond distance 1,
but the flag was False for each failure. The median number of iterations grows with the distance. For
`alpha = 0` on St(12, 4), it went from 3 at distance 0.5 to 11 at distance 2.5.

**Near the cut locus.** Take a frame of St(12, 4) and a point whose first column is the negative of that
of the frame. For this pair, the matrix logarithm has no real principal branch, and `log`
returned `converged = False` after 0 iterations, with residual 2. We then perturbed the point. It was
replaced by the polar factor of the point plus `eps` times a Gaussian matrix. With `eps = 1e-6` and with
`eps = 1e-3`, the iteration did not converge within `max_iter`, and the residual was of the size of `eps`. It
did converge for `eps = 0.1`, at a distance of about 2.8.

## Interoperability

**The `Stiefel` class** holds `n`, `p` and `alpha`, and it forwards to the functions under the method names
of pymanopt:

- `projection`, `inner_product` and `norm`;
- `exp`, `retraction` (QR, as in pymanopt) and `transport`;
- `log` (filled with NaN on failure), `log_result` (the full `LogResult`) and `dist`;
- `euclidean_to_riemannian_gradient`;
- `random_point`, `random_tangent_vector` and `zero_vector`.

**pymanopt.** `Stiefel(n, p, alpha=...).to_pymanopt(key)` returns a `pymanopt.manifolds.manifold.Manifold`
that converts numpy arrays in and out, so that pymanopt's optimizers run with any metric of the family:

```python
import jax
import numpy as np
import pymanopt

jax.config.update("jax_enable_x64", True)
import stiefel as sf

n, p = 200, 5
rng = np.random.default_rng(0)
a = rng.standard_normal((n, n))
a = a + a.T  # symmetric test matrix
manifold = sf.Stiefel(n, p).to_pymanopt(jax.random.key(0))  # canonical metric


@pymanopt.function.numpy(manifold)
def cost(x):
    return -np.trace(x.T @ a @ x)


@pymanopt.function.numpy(manifold)
def egrad(x):
    return -2 * a @ x


problem = pymanopt.Problem(manifold, cost, euclidean_gradient=egrad)
x = pymanopt.optimizers.ConjugateGradient(verbosity=0).run(problem).point
print(-cost(x), np.sort(np.linalg.eigvalsh(a))[-p:].sum())  # sum of the p largest eigenvalues
```

With pymanopt 2.2.1, this code prints `184.654308989375 184.6543089893757`. The adapter supports first-order
optimizers only, such as `SteepestDescent` and `ConjugateGradient`. It has no
`euclidean_to_riemannian_hessian`, so `TrustRegions` and the other second-order optimizers raise
`NotImplementedError` in pymanopt. The import of pymanopt switches on `jax_enable_x64`, and `to_pymanopt`
restores the flag as it found it. At `alpha = -1/2`, the tests check `projection`, `inner_product`, `exp` and
`retraction` against pymanopt's `Stiefel` to `1e-8`.

**geomstats.** At `alpha = 0`, `exp`, `log`, `inner` and `retract_qr` agree with geomstats 2.8.0 to `1e-8`.
The largest difference that we measured was `2.1e-13`, for a logarithm on St(6, 3) at distance 1.0. The
comparison uses arrays that geomstats computed in a separate environment with numpy 1.26
(`tools/geomstats_reference.py`, `tests/golden/geomstats.npz`). The reason is that geomstats 2.8 cannot be
imported under numpy 2.

In one of the six logarithm cases (St(3, 2) at distance 0.3), geomstats raised `ValueError`. When `n < 2p`,
the normal component of the point has rank below `p`. The QR factorization that geomstats takes of this
component then returns a basis that is not orthogonal to the base point. In that case, the test compares with
the tangent vector that generated the endpoint.

**Arrays.** Every function accepts numpy arrays and returns JAX arrays.

## Transformations, devices and precision

- `jit` and `vmap`: every function of `stiefel.geometry` works under both, that is `exp`, `log`, `distance`,
  `inner`, the retractions and the random draws. In the tests, `exp`, `log` and `distance` run under
  `jax.jit(jax.vmap(...))`. Because they are static arguments, `alpha`, `tol` and `max_iter` must be Python
  numbers. You cannot wrap `joint_basis`, `frechet_mean`, `pairwise_distance` or `kmeans` in `jit`, since
  they produce intermediate arrays whose shape depends on the data. Their inner loops are compiled.
- GPU: the functions are written so that they also run on a CUDA device (invariant S5). One test of the suite
  runs `exp` and `log` on such a device when `STIEFEL_TEST_GPU=1` is set. We did not run it for this release,
  because the GPU was busy with another job.
- Differentiation: `exp` is differentiable in forward and reverse mode at every tangent. This includes the zero
  tangent and a rank-deficient normal component (`n < 2p`). Its value is computed from the QR form of Edelman,
  Arias and Smith. For its derivative, which is defined with `jax.custom_jvp`, an equivalent form without QR
  is used. There, the normal component `K` enters as `K / c` and `K^T K / c`, with `c = max(|K|_F, 1)`. The
  entries of the exponent therefore grow as `|K|` and not as `|K|^2`. With `c = 1` instead, the derivative was NaN
  at `|K|_F = 1000`.

  We checked `jax.jvp` and `jax.grad` on St(10, 3) and St(5, 3), at the zero tangent and at a tangent of
  length 0.5. The derivatives were finite there. Also, the speed `norm(exp(x, t d), d/dt exp(x, t d))`
  equalled the length of `d` to `2e-16`. In the suite, both modes are compared with central differences at
  tangent lengths `1e-8`, 1, 30 and 300 for `alpha` in {-1/2, 0, 1}. One more comparison is made at `|K|_F = 1000`. In the
  domain of invariant S1 for `exp`, the derivative is finite.

  `log` raises `TypeError` when `jax.grad`, `jax.jvp` or another transformation differentiates through it. So
  does every function that calls it: `distance`, `pairwise_distance`, `frechet_mean`, `kmeans`, `Stiefel.log`
  and `Stiefel.dist`. The message of the error states that the logarithm is computed by an iteration. Implicit
  differentiation of the logarithm is not implemented.
- float32: the default `tol` of `log` is `1e-4`, and that of `frechet_mean` and `kmeans` is `1e-3`. A `tol`
  below `1.19e-5` raises `ValueError`. On St(10, 3), the logarithm converged at distances 0.5, 1 and 2, with
  errors below `4.4e-6`. The suite checks distance 0.5 to `1e-4`.
- Large tangents: `exp` returns a frame within the bound of invariant S1 when the exponents of its matrix
  exponentials are bounded. The bounds are `|A|_F / (1 + alpha) <= 1000`, `|alpha| |A|_F / (1 + alpha) <= 1000`
  and `|K|_F <= 500`, with `A = skew(base^T tangent)` and `K` the normal component. At the corner of this
  domain, we measured at most 0.84 of the bound. For `-1/2 <= alpha <= 1`, this domain includes every tangent
  of Riemannian norm up to 500.

  For other `alpha`, the Riemannian norm does not bound the exponents. At `alpha = 1e16`, a tangent of norm 1
  can give `|alpha| |A|_F / (1 + alpha) = 1.4e8`. The result is then NaN, because `jax.scipy.linalg.expm`
  returns NaN once the 1-norm of its argument reaches about `7e5` in float64. Beyond the domain, the error of
  orthonormality grows with the norms. `exp` raises `ValueError` for an `alpha` that is not finite or is at
  most -1.

## How it is specified and tested

- **Contracts.** Each public function has a docstring with the sections `Requires:` and `Ensures:`. The
  caller must pass what `Requires:` states. In `Ensures:`, you find what the function returns, what it costs,
  and what it does on failure.
- **Invariants.** `INVARIANTS.md` states ten properties, each of which spans several functions. Docstrings
  cite them as S1 to S10:
  - S1, frames stay frames;
  - S2, no unvalidated logarithm;
  - S3, exact compression;
  - S4, precision;
  - S5, transformations and devices;
  - S6, cost;
  - S7, explicit randomness;
  - S8, library hygiene;
  - S9, agreement with other packages;
  - S10, own code.
- **Property tests.** Hypothesis draws shapes that include `n = p`, `n < 2p` and `p = 1`. Among other things,
  the tests check the following:
  - the outputs are frames;
  - `exp` agrees with the formula of Edelman, Arias and Smith, evaluated with SciPy;
  - the round trip `log(base, exp(base, D)) = D` holds;
  - failures are reported.

  During the tests, beartype checks the jaxtyping annotations of the library at run time.
- **Reference arrays.** `tests/golden/exp_log.npz` holds outputs of an earlier implementation of the same
  algorithms. The outputs of geomstats are in `tests/golden/geomstats.npz`, and `tests/golden/README.md`
  describes both files.
- **JAX versions.** On the CPU with Python 3.13, the full suite passes with jax 0.12.0.1 and with jax 0.7.0.
  With jax 0.12.0.1, on 11 November 2026, 302 tests passed and 7 were skipped. Of the skipped tests, two are
  the CUDA tests and four are golden cases in which the earlier implementation did not converge. The last one
  is the direct geomstats test.

  With jax 0.7.0, 270 tests passed in our shared environment, where `tests/test_hygiene.py` failed. It failed
  only because a CUDA plugin of jax 0.12 was installed there and wrote a warning to stderr. In a clean
  environment with jax 0.7.0, this test passes.

  With jax 0.8.0, 0.9.0 and 0.10.0, we ran 158 tests: those of `exp` and `log` in `tests/test_geometry.py`
  and those of `tests/test_golden.py`. They also pass. We therefore require `jax>=0.7`, and CI checks this
  requirement with jax 0.7.0.

Run the suite and the notebook check with these commands:

```
uv run pytest                                                   # the suite, on the CPU
uv run --with jupyterlab --with matplotlib --with nbmake pytest --nbmake examples/
```

## References

No code of another project is copied (invariant S10), although the algorithms follow the papers below. The
arXiv numbers that we give are those of the versions that we read; the journal data belong to the published
versions.

- A. Edelman, T. A. Arias and S. T. Smith. The geometry of algorithms with orthogonality constraints.
  *SIAM Journal on Matrix Analysis and Applications* 20(2):303–353, 1998. arXiv:physics/9806030. Its
  Corollary 2.2 gives the closed-form exponential of the canonical metric.
- R. Zimmermann. A matrix-algebraic algorithm for the Riemannian logarithm on the Stiefel manifold under the
  canonical metric. *SIAM Journal on Matrix Analysis and Applications* 38(2):322–342, 2017.
  arXiv:1604.05054. From it come the algebraic logarithm and its local linear convergence.
- R. Zimmermann and K. Hüper. Computing the Riemannian logarithm on the Stiefel manifold: metrics, methods and
  performance. *SIAM Journal on Matrix Analysis and Applications* 43(2):953–980, 2022. arXiv:2103.12046. We
  take from it the metric family, its exponential, and the Sylvester-accelerated logarithm.
- S. Mataigne, R. Zimmermann and N. Miolane. An efficient algorithm for the Riemannian logarithm on the Stiefel
  manifold for a family of Riemannian metrics. arXiv:2403.11730v2, 2024. This paper gives the logarithm for
  every metric of the family, and its Appendix B gives the initial completion.
