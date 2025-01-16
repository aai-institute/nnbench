import numpy as np
import pytest

import nnbench

N = 1024
NUM_REPLICAS = 3
NP_MATSIZE_BYTES = N**2 * np.float64().itemsize
# for a matmul, we alloc LHS, RHS, and RESULT.
# math.ceil gives us a cushion of up to 1MiB, which is for other system allocs.
EXPECTED_MEM_USAGE_MB = 3 * NP_MATSIZE_BYTES / 1048576 + 0.5


@pytest.mark.limit_memory(f"{EXPECTED_MEM_USAGE_MB}MB")
def test_parametrize_memory_consumption():
    """
    Checks that a benchmark family works with GC in the parametrization case,
    and produces the theoretically optimal memory usage pattern for a matmul.

    Note: We do not have a similar "best case" memory guarantee for @nnbench.product,
    because the evaluation of the cartesian product via `itertools.product()` forces
    the eager exhaustion of all iterables (generators can be used only once).
    """

    @nnbench.parametrize({"b": np.zeros((N, N), dtype=np.int64)} for _ in range(NUM_REPLICAS))
    def matmul(a: np.ndarray, b: np.ndarray) -> np.float64:
        return np.dot(a, b).sum()

    a = np.zeros((N, N), dtype=np.int64)
    nnbench.run(matmul, params={"a": a})
