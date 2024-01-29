import os

from nnbench.runner import BenchmarkRunner


def test_runner_discovery(testfolder: str) -> None:
    r = BenchmarkRunner()

    r.collect(os.path.join(testfolder, "standard_benchmarks.py"), tags=("runner-collect",))
    assert len(r.benchmarks) == 1

    r.clear()

    r.collect(testfolder, tags=("runner-collect",))
    assert len(r.benchmarks) == 1


def test_tag_selection(testfolder: str) -> None:
    PATH = os.path.join(testfolder, "tag_selection_benchmark.py")

    r = BenchmarkRunner()

    r.collect(PATH, tags=())
    assert len(r.benchmarks) == 3
    r.clear()

    r.collect(PATH, tags=("tag1",))
    assert len(r.benchmarks) == 2
    r.clear()

    r.collect(PATH, tags=("tag2",))
    assert len(r.benchmarks) == 1
    r.clear()
