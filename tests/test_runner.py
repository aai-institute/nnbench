import os

from mlbench.runner import AbstractBenchmarkRunner


def test_runner_discovery(testfolder: str) -> None:
    r = AbstractBenchmarkRunner()

    r.collect(os.path.join(testfolder, "hello.py"))
    assert len(r.benchmarks) == 1

    r.clear()

    r.collect(testfolder)
    assert len(r.benchmarks) == 1
