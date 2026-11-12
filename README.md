# stiefel

Riemannian geometry and statistics on the Stiefel manifold, written in JAX.
A point of St(n, p) is an n-by-p matrix with orthonormal columns, such as a reduced basis.

- Geodesics, logarithms, distances and retractions for a family of metrics that includes the canonical and Euclidean metrics.
- Fréchet means and k-means, with convergence reports.
- Compression to a joint subspace, so that statistics on large frames can be computed in smaller coordinates.
- A pymanopt adapter for first-order optimisation with any metric of the family.

## Installation

Requires Python 3.11 or later and JAX 0.7 or later. Install from GitHub:

```sh
pip install "stiefel @ git+https://github.com/jinyi12/stiefel.git"
```

The `compat` extra installs pymanopt; the `gpu` extra installs JAX for CUDA 12.
For example, use `stiefel[compat]` in place of `stiefel` in the command above.

## Quickstart

```python
import jax
import jax.numpy as jnp
import stiefel as sf

jax.config.update("jax_enable_x64", True)

kx, kd = jax.random.split(jax.random.key(0))
x = sf.random_point(kx, 50, 4)
d = sf.random_tangent(kd, x)
d = 0.5 * d / sf.norm(x, d)
y = sf.exp(x, d)

result = sf.log(x, y)
print(bool(result.converged), float(result.residual))
print(float(jnp.max(jnp.abs(result.tangent - d))))
```

The default metric is canonical (`alpha=0`). Pass `alpha=-0.5` for the Euclidean metric;
the family is defined for `alpha > -1`. The library preserves the precision of its inputs.
Enable float64 before creating arrays, as above, for the accuracy used in the tests.

## Numerical behaviour

The logarithm is computed by a local iteration. Check `result.converged` before using its tangent;
`distance` returns NaN when the logarithm fails. A converged result need not be the shortest geodesic
outside the injectivity radius. Fréchet means and k-means are also local methods.

The geometry functions support `jax.jit` and `jax.vmap`. The exponential supports differentiation;
the logarithm and functions that call it do not. Compression and statistics determine the width of
the joint basis from the data and cannot be wrapped in `jit` as a whole.

## Examples and documentation

- [Example notebook](examples/quickstart.ipynb): geodesics, compression, Fréchet means and clustering.
- [Numerical notes](docs/numerical-notes.md): metric definitions, algorithms, convergence limits and references.
- [Function docstrings](src/stiefel/geometry.py) and [invariants](INVARIANTS.md): input requirements and guarantees.
- [Benchmarks](benchmarks/results.md): CPU timings and measurement conditions.

## Development

From a checkout, install the development dependencies and run the tests with [uv](https://docs.astral.sh/uv/):

```sh
git clone https://github.com/jinyi12/stiefel.git
cd stiefel
uv sync
uv run pytest
```

The tests run on the CPU by default. Set `STIEFEL_TEST_GPU=1` to include the CUDA checks.

## Licence and citation

[BSD 3-Clause](LICENSE). See [AUTHORS.md](AUTHORS.md) for authorship and [CITATION.cff](CITATION.cff)
for citation metadata. The algorithm references are listed in the [numerical notes](docs/numerical-notes.md#references).
