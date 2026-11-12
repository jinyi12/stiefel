# Fast intrinsic clustering

For a collection of Stiefel frames, `emeans` seeks a partition whose centres minimize
the sum of squared geodesic distances. It uses mini-batches to update the centres before
refining their intrinsic means. Thus the objective is the same as for `kmeans`,

$$
F(C_1,\ldots,C_k)=\sum_i\min_j d_\alpha(X_i,C_j)^2.
$$

```python
result = sf.emeans(key, frames, n_clusters=8, alpha=0.0)
print(result.converged, result.inertia, result.log_evaluations)
```

The return value is an `EMeansResult`. In particular, `iterations` gives the number of training
epochs attempted, so an epoch interrupted by a numerical failure is counted as well.
The algorithms and defaults of `kmeans` and `frechet_mean` are retained; their mean solves
now omit observations that have zero weight.

## Updates and assignment bounds

For the points in a mini-batch assigned to centre $C_j$, let $m_j$ be their count and let
$G_j=\sum_{i\in B_j}\operatorname{Log}_{C_j}(X_i)$. Denote the decayed count for this
centre by $N_j$. With step $s_t$, the update is

$$
N_j^+=(1-s_t)N_j+s_tm_j,\qquad
C_j^+=\operatorname{Exp}_{C_j}\!\left(\frac{s_t}{N_j^+}G_j\right).
$$

If no point is assigned to a centre in the batch, that centre is left unchanged.
In Euclidean space the formula reduces to an exponential moving average (EMA) of sums
and counts. For a curved manifold, however, one cannot recover the exact Fréchet mean
of the weighted history from the current centre and count alone.

Since a logarithm used for assignment gives both a distance and a tangent vector, it can
also be used in the update. Some candidate centres can be excluded without computing
their logarithms. To obtain a bound for this purpose, set $\beta=1/[2(1+\alpha)]$ and
observe that the metric satisfies

$$
g_X(V,V)\geq\min(1,\beta)\|V\|_F^2,
\quad\text{hence}\quad
d_\alpha(X,C)\geq\sqrt{\min(1,\beta)}\|X-C\|_F.
$$

Candidates are considered in order of increasing lower bound. Suppose that a logarithm
has been computed for one centre and that the resulting path is shorter than the bound
for another centre. The latter cannot improve the assignment and need not be evaluated.
Rounding and the tolerance on the endpoint residual are allowed for in this comparison.
The path obtained by the logarithm is guaranteed to be minimal only within its local
domain, as with the other distance computations in the package.

For initialization, five starts are compared using the ambient objective, so that no
logarithms are needed at this stage. Each start consists of ambient k-means++ sampling
followed by five projected-mean iterations. The step, initially `1 - decay`, is decreased
to one tenth of its initial value along a cosine schedule. After every five batches,
a cluster with little mass can be revived at a poorly represented point of the current
batch, with mass transferred from a donor. Meta instead uses a Euclidean splitting rule.

Once training has finished, each centre is refined with at most `polish_iter` intrinsic
Fréchet iterations, using only the observations assigned to it. In this way a mean solve
is carried out at the end, whereas Lloyd's method repeats mean solves as its assignments
change. The Lloyd implementation also restricts each solve to the assigned members.
For e-means, a further check of assignments and mean tangents is made at the final centres.

## Accuracy and cost

A result is marked `converged` only if the logarithms were successful, the final assignments
were checked, and the mean tangent in every cluster has norm at most `tol`. If refinement
is insufficient, or if the assignments have changed, the inertia may be finite although
`converged` is false. When an assignment cannot be trusted, the inertia is NaN.
Empty clusters are repaired, including those caused by ties between identical frames,
and the convergence decision takes that repair into account.

The tolerance defaults to `1e-6` for float64 and `1e-3` for float32. Further refinement
can be requested through `polish_iter` or `epochs` if the final check fails.
Batch size and decay will also need to be chosen for the application, since the scaling
laws fitted by Meta in Euclidean space have not been validated for this geometry.
During the stochastic phase the loss may increase, and a global optimum is not guaranteed.

When there are fewer stacked columns than ambient dimensions, a joint basis is computed
once for the ensemble. Otherwise the frames are used directly, and an ambient identity
matrix need not be allocated. A saving from compression is possible only to the extent
that the joint subspace is small. The data and their compressed coordinates must fit
in memory: this interface does not read an out-of-core stream. Although assignments are
processed in batches, the final checks retain the winning tangent for every observation.

Run the CPU comparison with

```sh
uv run python benchmarks/clustering.py
```

For each solver, the script times a first call and a repeated call on the same data.
It also reports intrinsic inertia and convergence, together with the number of logarithms
evaluated by `emeans`. The tolerance is held fixed, while each solver uses its own
initialization. When assessing the cost of a single use, compilation and initialization
must therefore be included. The measured cases and their restrictions are given in the
[benchmark results](../benchmarks/results.md).

## Research used in the implementation

The search covered alphaXiv and OpenAlex through 12 November 2026. For the present
implementation, the relevant distinction is between reducing the cost of an intrinsic
mean solve and replacing that solve with an average for a different objective.

A starting centre can be obtained by taking the polar projection of an arithmetic average.
This is the Stiefel average obtained from retraction-lifting barycenters by
[Bouchard, Laurent, Said and Le Bihan (2025), sections II–III](https://www.alphaxiv.org/pdf/2501.11555?page=2).
Since it requires no logarithms, we use it both to initialize `frechet_mean` and to compare
inexpensive starts for `emeans`. An intrinsic refinement is still needed if stationarity
for the geodesic objective is required. The projected average generally differs from the
intrinsic Fréchet mean.

The use of a cheaper objective has also been studied by
[Mataigne, Absil and Miolane (2025), sections 3–5](https://www.alphaxiv.org/pdf/2504.15671?page=3).
Their bound concerns the objective error at a global minimizer of the surrogate; an
arbitrary iterate produced by a numerical solver is consequently not certified by that
bound. For pruning assignments here, the comparison of metrics gives the lower bound
stated above. In particular, it can be used without the assumption that the ambient
dimension is at least twice the number of columns, which is needed for the stronger
Stiefel distance bounds in that work.

Repeated geodesic solves can instead be avoided by optimizing the geodesics together
with the mean. This is the approach of GEORCE-FM, formulated in local coordinates by
[Rygaard, Hauberg and Markvorsen (2025)](https://www.alphaxiv.org/pdf/2511.04301).
It would require a different mean solver and has not been implemented in this package.

For clustering, a change of representation must also be distinguished from a faster solve
on the original manifold. [Shi, Charon, Mang, Labate and Azencott (2025), sections 1–2](https://www.alphaxiv.org/pdf/2511.08993?page=1)
apply Euclidean clustering to representations obtained from Fréchet maps, with experiments
on symmetric positive-definite matrices. Whether this representation is suitable for
Stiefel observations would have to be assessed separately. In the
[2026 paper on Riemannian difference-of-convex clustering, section 2](https://www.alphaxiv.org/pdf/2609.34310?page=2),
Stiefel constraints are imposed on an assignment matrix, while the observations themselves
are Euclidean. That problem therefore differs from grouping the frames considered here.

The decayed sums and counts, cosine schedule, cluster revival and final mean updates are
parts of the Euclidean e-means algorithm of
[Bessa et al. (2026), sections 3.1–3.3](https://www.alphaxiv.org/pdf/2611.09501?page=3).
Here the count normalization and schedule are used with tangent vectors at the current
centres. Online updates using exponential and logarithm maps were already considered in
competitive learning Riemannian quantization by
[Le Brigant and Puechmorel](https://www.alphaxiv.org/pdf/1806.07605).
The resulting adaptation uses that form of update with decayed counts, followed by a
bounded intrinsic refinement.
