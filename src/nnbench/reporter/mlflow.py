import os
from contextlib import ExitStack
from pathlib import Path
from typing import TYPE_CHECKING, Any

from nnbench.types import BenchmarkReporter, BenchmarkResult

if TYPE_CHECKING:
    from mlflow import ActiveRun as ActiveRun


class MLFlowReporter(BenchmarkReporter):
    def __init__(self):
        self.stack = ExitStack()

    @staticmethod
    def strip_protocol(uri: str | os.PathLike[str]) -> str:
        s = str(uri)
        if s.startswith("mlflow://"):
            return s[9:]
        return s

    def get_or_create_run(self, run_name: str, nested: bool = False) -> "ActiveRun":
        import mlflow

        existing_runs = mlflow.search_runs(
            filter_string=f"attributes.`run_name`={run_name!r}", output_format="list"
        )
        if existing_runs:
            run_id = existing_runs[0].info.run_id
            return mlflow.start_run(run_id=run_id, nested=nested)
        else:
            return mlflow.start_run(run_name=run_name, nested=nested)

    def read(self, uri: str | os.PathLike[str], **kwargs: Any) -> list[BenchmarkResult]:
        raise NotImplementedError

    def write(
        self,
        result: BenchmarkResult,
        uri: str | os.PathLike[str],
        **kwargs: Any,
    ) -> None:
        import mlflow

        uri = self.strip_protocol(uri)
        try:
            experiment, run_name, *subruns = Path(uri).parts
        except ValueError:
            raise ValueError(f"expected URI of form <experiment>/<run>[/<subrun>...], got {uri!r}")

        # setting experiment removes the need for passing the `experiment_id` kwarg
        # in the subsequent API calls.
        mlflow.set_experiment(experiment)

        run = self.stack.enter_context(self.get_or_create_run(run_name=run_name))
        for s in subruns:
            # reassignment ensures that we log into the max-depth subrun specified.
            run = self.stack.enter_context(self.get_or_create_run(run_name=s, nested=True))

        run_id = run.info.run_id
        for res in result:
            timestamp = res.timestamp
            mlflow.log_dict(res.context, f"context-{res.run}.json", run_id=run_id)
            for bm in res.benchmarks:
                name, value = bm["name"], bm["value"]
                mlflow.log_metric(name, value, timestamp=timestamp, run_id=run_id)
