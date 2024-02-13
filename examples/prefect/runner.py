from typing import Any

import numpy as np
import training
from prefect import flow, get_run_logger, task
from sklearn import base

import nnbench
from nnbench import reporter, types


class PrefectReporter(reporter.BenchmarkReporter):
    def __init__(self):
        self.logger = get_run_logger()

    def write(self, record: types.BenchmarkRecord, **kwargs: Any) -> None:
        self.logger.info(record)


@task
def run_metric_benchmarks(
    model: base.BaseEstimator, X_test: np.ndarray, y_test: np.ndarray
) -> nnbench.types.BenchmarkRecord:
    runner = nnbench.BenchmarkRunner()
    results = runner.run(
        "benchmark.py",
        tags=("metric",),
        params={"model": model, "X_test": X_test, "y_test": y_test},
    )
    return results


@task
def run_metadata_benchmarks(
    model: base.BaseEstimator, X: np.ndarray
) -> nnbench.types.BenchmarkRecord:
    runner = nnbench.BenchmarkRunner()
    result = runner.run("benchmark.py", tags=("model-meta",), params={"model": model, "X": X})
    return result


@flow
def train_and_benchmark(random_state: int = 42) -> tuple[types.BenchmarkRecord, ...]:
    reporter = PrefectReporter()

    model, X_test, y_test = training.prepare_regressor_and_test_data(random_state=random_state)

    metadata_results = run_metadata_benchmarks(model=model, X=X_test)
    reporter.write(metadata_results)

    metric_results = run_metric_benchmarks(model=model, X_test=X_test, y_test=y_test)
    reporter.write(metric_results)

    return metadata_results, metric_results


if __name__ == "__main__":
    train_and_benchmark.serve(name="benchmark-runner")
