from __future__ import annotations

import asyncio
import os

import numpy as np
import training
from prefect import flow, get_run_logger, task
from prefect.artifacts import create_table_artifact
from sklearn import base

import nnbench
from nnbench import context, reporter, types

dir_path = os.path.dirname(__file__)


class PrefectReporter(reporter.BenchmarkReporter):
    def __init__(self):
        self.logger = get_run_logger()

    async def write(
        self, record: types.BenchmarkRecord, key: str, description: str = "Benchmark and Context"
    ) -> None:
        await create_table_artifact(
            key=key,
            table=record.compact(mode="flatten"),
            description=description,
        )


@task
def run_metric_benchmarks(
    model: base.BaseEstimator, X_test: np.ndarray, y_test: np.ndarray
) -> nnbench.types.BenchmarkRecord:
    runner = nnbench.BenchmarkRunner()
    results = runner.run(
        os.path.join(dir_path, "benchmark.py"),
        tags=("metric",),
        params={"model": model, "X_test": X_test, "y_test": y_test},
    )
    return results


@task
def run_metadata_benchmarks(
    model: base.BaseEstimator, X: np.ndarray
) -> nnbench.types.BenchmarkRecord:
    runner = nnbench.BenchmarkRunner()
    result = runner.run(
        os.path.join(dir_path, "benchmark.py"),
        tags=("model-meta",),
        params={"model": model, "X": X},
    )
    return result


@flow(persist_result=True)
async def train_and_benchmark(
    data_params: dict[str, int | float] | None = None,
) -> tuple[types.BenchmarkRecord, ...]:
    if data_params is None:
        data_params = {}

    reporter = PrefectReporter()

    regressor_and_test_data: tuple[
        base.BaseEstimator, np.ndarray, np.ndarray
    ] = await training.prepare_regressor_and_test_data(data_params=data_params)  # type: ignore

    model = regressor_and_test_data[0]
    X_test = regressor_and_test_data[1]
    y_test = regressor_and_test_data[2]

    metadata_results: types.BenchmarkRecord = run_metadata_benchmarks(model=model, X=X_test)

    metadata_results.context.update(data_params)
    metadata_results.context.update(context.PythonInfo())

    await reporter.write(
        record=metadata_results, key="model-attributes", description="Model Attributes"
    )

    metric_results: types.BenchmarkRecord = run_metric_benchmarks(
        model=model, X_test=X_test, y_test=y_test
    )

    metric_results.context.update(data_params)
    metric_results.context.update(context.PythonInfo())
    await reporter.write(metric_results, key="model-performance", description="Model Performance")
    return metadata_results, metric_results


if __name__ == "__main__":
    asyncio.run(train_and_benchmark.serve(name="benchmark-runner"))
