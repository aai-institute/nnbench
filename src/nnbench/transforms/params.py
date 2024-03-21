from typing import Any, Sequence

from nnbench.transforms import ManyToManyTransform, OneToOneTransform
from nnbench.types import BenchmarkRecord


class CompressionMixin:
    def compress(self, params: dict[str, Any]) -> dict[str, Any]:
        containers = (tuple, list, set, frozenset)
        natives = (float, int, str, bool, bytes, complex)
        compressed: dict[str, Any] = {}

        def _compress_impl(val):
            if isinstance(val, natives):
                # save native types without modification...
                return val
            else:
                # ... or return the string repr.
                # TODO: Allow custom representations for types with formatters.
                return repr(val)

        for k, v in params.items():
            if isinstance(v, containers):
                container_type = type(v)
                compressed[k] = container_type(_compress_impl(vv) for vv in v)
            elif isinstance(v, dict):
                compressed[k] = self.compress(v)
            else:
                compressed[k] = _compress_impl(v)

        return compressed


class ParameterCompression1to1(OneToOneTransform, CompressionMixin):
    def apply(self, record: BenchmarkRecord) -> BenchmarkRecord:
        for bm in record.benchmarks:
            bm["params"] = self.compress(bm["params"])

        return record


class ParameterCompressionNtoN(ManyToManyTransform, CompressionMixin):
    def apply(self, record: Sequence[BenchmarkRecord]) -> Sequence[BenchmarkRecord]:
        for rec in record:
            for bm in rec.benchmarks:
                bm["params"] = self.compress(bm["params"])

        return record
