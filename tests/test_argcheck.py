import logging
import os

import pytest

from nnbench import runner


def test_argcheck(testfolder: str) -> None:
    benchmarks = os.path.join(testfolder, "benchmarks.py")
    r = runner.AbstractBenchmarkRunner()
    with pytest.raises(TypeError, match="expected type <class 'int'>.*"):
        r.run(benchmarks, params={"x": 1, "y": "1"})
    with pytest.raises(ValueError, match="missing value for required parameter.*"):
        r.run(benchmarks, params={"x": 1})

    r.run(benchmarks, params={"x": 1, "y": 1})


def test_error_on_duplicate_params(testfolder: str) -> None:
    benchmarks = os.path.join(testfolder, "duplicate_benchmarks.py")

    with pytest.raises(TypeError, match="got incompatible types.*"):
        r = runner.AbstractBenchmarkRunner()
        r.run(benchmarks, params={"x": 1, "y": 1})


def test_log_warn_on_overwrite_default(
    testfolder: str, caplog: pytest.LogCaptureFixture
) -> None:
    benchmark = os.path.join(testfolder, "default_benchmarks.py")
    r = runner.AbstractBenchmarkRunner()
    with caplog.at_level(logging.DEBUG):
        r.run(benchmark, params={"a": 1})
    assert "using given value 1 over default value" in caplog.text


def test_untyped_interface(testfolder: str) -> None:
    benchmarks = os.path.join(testfolder, "untyped_benchmarks.py")

    r = runner.AbstractBenchmarkRunner()
    r.run(benchmarks, params={"value": 2})
