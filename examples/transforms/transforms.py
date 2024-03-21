from dataclasses import dataclass
from typing import Any

import numpy as np

import nnbench
from nnbench.reporter.file import FileIO
from nnbench.transforms import OneToOneTransform
from nnbench.types import BenchmarkRecord


class MyModel:
    def __init__(self, checksum: str):
        self.checksum = checksum

    def apply(self, data: np.ndarray) -> float:
        return data.mean()

    def to_json(self) -> dict[str, Any]:
        return {"checksum": self.checksum}

    @classmethod
    def from_json(cls, obj: dict[str, Any]) -> "MyModel":
        # intentionally fail if no checksum is given.
        return cls(checksum=obj["checksum"])


@nnbench.benchmark
def accuracy(model: MyModel, data: np.ndarray) -> float:
    return model.apply(data)


class MyTransform(OneToOneTransform):
    def apply(self, record: BenchmarkRecord) -> BenchmarkRecord:
        """Apply this transform on a record."""
        for b in record.benchmarks:
            params: dict[str, Any] = b["parameters"]
            b["parameters"] = {
                "model": params["model"].to_json(),
                "data": params["data"].tolist(),
            }
        return record

    def iapply(self, record: BenchmarkRecord) -> BenchmarkRecord:
        """Apply the inverse of this transform."""
        for b in record.benchmarks:
            params: dict[str, Any] = b["parameters"]
            b["parameters"] = {
                "model": MyModel.from_json(params["model"]),
                "data": np.asarray(params["data"]),
            }
        return record


def main():
    @dataclass(frozen=True)
    class MyParams(nnbench.Parameters):
        model: MyModel
        data: np.ndarray

    runner = nnbench.BenchmarkRunner()

    m = MyModel(checksum="12345")
    data = np.random.random_sample((10,))
    params = MyParams(m, data)
    record = runner.run(__name__, params=params)

    transform = MyTransform()
    trecord = transform.apply(record)
    f = FileIO()
    f.write(trecord, "record.json")

    record2 = f.read("record.json")
    new_record = transform.iapply(record2)


if __name__ == "__main__":
    main()
