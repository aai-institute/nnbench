import os

from nnbench.runner import AbstractBenchmarkRunner


def test_runner_discovery(testfolder: str) -> None:
    r = AbstractBenchmarkRunner()

    r.collect(os.path.join(testfolder, "standard_benchmarks.py"), tags=("runner-collect",))
    assert len(r.benchmarks) == 1

    r.clear()

    r.collect(testfolder, tags=("runner-collect",))
    assert len(r.benchmarks) == 1
