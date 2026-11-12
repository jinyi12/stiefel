import os

import jax
import pytest
from jaxtyping import install_import_hook

# The tests run on the CPU, because concurrent runs on the shared GPU ran out of memory. With
# STIEFEL_TEST_GPU=1, the tests run on the default device instead, and the CUDA smoke test is included.
if os.environ.get("STIEFEL_TEST_GPU") != "1":
    jax.config.update("jax_platforms", "cpu")

# The tests run in float64 (S4). The library never changes this setting itself.
jax.config.update("jax_enable_x64", True)

# Check every jaxtyping annotation of the library at run time. The hook decorates the functions of the
# package when the package is imported, so that the library itself does not depend on beartype.
with install_import_hook("stiefel", "beartype.beartype"):
    import stiefel  # noqa: F401


@pytest.fixture(autouse=True, scope="module")
def _free_compiled_code():
    """Drop compiled executables after each module.

    The suite compiles thousands of small executables, for many shapes, metrics and tolerances. Without this
    step, a single process would exhaust the limit on memory mappings (vm.max_map_count) inside LLVM.
    """
    yield
    jax.clear_caches()
