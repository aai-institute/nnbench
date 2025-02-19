import os
from contextlib import ExitStack
from datetime import datetime
from pathlib import Path
from typing import TYPE_CHECKING, Any, Protocol

from nnbench.reporter.util import get_protocol
from nnbench.types import BenchmarkRecord

if TYPE_CHECKING:
    from mlflow import ActiveRun as ActiveRun


class BenchmarkServiceIO(Protocol):
    def read(self, uri: str | os.PathLike[str], options: dict[str, Any]) -> BenchmarkRecord: ...

    def write(
        self, record: BenchmarkRecord, uri: str | os.PathLike[str], options: dict[str, Any]
    ) -> None: ...


class MLFlowIO(BenchmarkServiceIO):
    def __init__(self):
        import mlflow

        self.mlflow = mlflow
        self.stack = ExitStack()

    @staticmethod
    def strip_protocol(uri: str | os.PathLike[str]) -> str:
        s = str(uri)
        if s.startswith("mlflow://"):
            return s[9:]
        return s

    def get_or_create_run(self, run_name: str, nested: bool = False) -> "ActiveRun":
        existing_runs = self.mlflow.search_runs(
            filter_string=f"attributes.`run_name`={run_name!r}", output_format="list"
        )
        if existing_runs:
            run_id = existing_runs[0].info.run_id
            return self.mlflow.start_run(run_id=run_id, nested=nested)
        else:
            return self.mlflow.start_run(run_name=run_name, nested=nested)

    def read(self, uri: str | os.PathLike[str], options: dict[str, Any]) -> BenchmarkRecord:
        raise NotImplementedError

    def write(
        self, record: BenchmarkRecord, uri: str | os.PathLike[str], options: dict[str, Any]
    ) -> None:
        uri = self.strip_protocol(uri)
        try:
            experiment, run_name, *subruns = Path(uri).parts
        except ValueError:
            raise ValueError(f"expected URI of form <experiment>/<run>[/<subrun>]..., got {uri!r}")

        # setting experiment removes the need for passing the `experiment_id` kwarg
        # in the subsequent API calls.
        self.mlflow.set_experiment(experiment_name=experiment)

        run = self.stack.enter_context(self.get_or_create_run(run_name=run_name))
        for s in subruns:
            # reassignment ensures that we log into the max-depth subrun specified.
            run = self.stack.enter_context(self.get_or_create_run(run_name=s, nested=True))

        self.mlflow.log_dict(record.context, "context.json", run_id=run.info.run_id)
        for bm in record.benchmarks:
            name, value = bm["name"], bm["value"]
            timestamp = int(datetime.fromisoformat(bm["date"]).timestamp())
            self.mlflow.log_metric(name, value, timestamp=timestamp, run_id=run.info.run_id)


# TODO: Investigate if this can be merged with the file io mapping to a top-level struct
#  (e.g. with a value union type BenchmarkFileIO | BenchmarkServiceIO | ...)
_service_io_mapping: dict[str, type[BenchmarkServiceIO]] = {
    "mlflow": MLFlowIO,
}


def get_service_io_class(uri: str) -> BenchmarkServiceIO:
    proto = get_protocol(uri)
    try:
        return _service_io_mapping[proto]()
    except KeyError:
        raise ValueError(f"unsupported benchmark IO: {proto!r}") from None


def register_file_io_class(
    name: str, klass: type[BenchmarkServiceIO], clobber: bool = False
) -> None:
    if name in _service_io_mapping and not clobber:
        raise RuntimeError(
            f"IO {name!r} is already registered (to force registration, rerun with clobber=True)"
        )
    _service_io_mapping[name] = klass
