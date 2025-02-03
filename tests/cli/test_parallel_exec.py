import time
from pathlib import Path

from nnbench.cli import main

DELAY_SECONDS = 10


def test_parallel_execution_for_slow_benchmarks():
    """
    Verifies that benchmarks with long execution time finish faster
    when using process parallelism.
    All discovered benchmarks contain a time.sleep(DELAY_SECONDS) call,
    and are just a simple add/mul of two integers, so we expect the
    execution time to be slightly above DELAY_SECONDS.
    """
    n_jobs = 2
    bm_path = Path(__file__).parent / "benchmarks"
    start = time.time()
    args = ["run", f"{bm_path}", "-j2"]
    rc = main(args)
    end = time.time() - start
    assert rc == 0, f"running nnbench {' '.join(args)} failed with exit code {rc}"
    assert end - start < n_jobs * DELAY_SECONDS
