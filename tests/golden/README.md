# Golden arrays

`exp_log.npz` holds reference values for `tests/test_golden.py`. The values were computed by the legacy Stiefel
routines of the stochastic-rom research code: `Stiefel_Exp_Log.py` and `Stiefel_Aux.py`, which are
R. Zimmermann's implementation of Zimmermann and Hüper 2022. The cases are the following:

- (n, p) in {(12, 3), (50, 5), (5, 3)};
- alpha in {0, -0.5, 1};
- tangent vectors of canonical norm 0.1, 1.0 and 1.5 at a random base point.

For each case, the file stores the exponential and the distance. It also stores the logarithm of the endpoint
with tau = 1e-11, its round-trip residual, and the alpha inner product of two tangent vectors. `{key}_legacy_ok` marks the cases in
which the legacy logarithm met the endpoint to 1e-8. The tests skip the other cases, because the shooting method
for alpha != 0 does not converge for n < 2p. The array `provenance` names the legacy function of every array and
the date of the legacy reference snapshot.

The file was made in the stochastic-rom repository with

```
python tools/make_golden.py --out-stiefel ../stiefel/tests/golden
```

The legacy environment is listed in its `tools/golden_environment.txt`. The legacy code itself is not
distributed with this repository; only the arrays that it produced are.

`geomstats.npz` holds the outputs of geomstats 2.8.0 (`StiefelCanonicalMetric.exp`, `.log`, `.inner_product`,
`.retraction`) for (n, p) in {(3, 2), (6, 3), (12, 4)}. The outputs were made with `tools/geomstats_reference.py`
in a numpy 1.26 environment (see its docstring), and `tests/test_geomstats_reference.py` uses them.
`{key}_log_ok` marks the one case (St(3, 2), distance 0.3) in which geomstats' logarithm raised an error. In
that case, the test compares with the tangent vector that generated the endpoint instead. `provenance` lists
the versions and the calls.
