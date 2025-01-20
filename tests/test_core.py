import nnbench
import nnbench.types
from nnbench import benchmark, parametrize, product

from .test_utils import has_expected_args


def test_benchmark_no_args():
    @benchmark
    def sample_benchmark() -> str:
        return "test"

    assert isinstance(sample_benchmark, nnbench.types.Benchmark)


def test_benchmark_with_args():
    @benchmark(name="Test Name", tags=("tag1", "tag2"))
    def another_benchmark() -> str:
        return "test"

    assert another_benchmark.name == "Test Name"
    assert another_benchmark.tags == ("tag1", "tag2")


def test_parametrize():
    @parametrize([{"param": 1}, {"param": 2}])
    def parametrized_benchmark(param: int) -> int:
        return param

    all_benchmarks = list(parametrized_benchmark)
    assert len(all_benchmarks) == 2
    assert has_expected_args(all_benchmarks[0].fn, {"param": 1})
    assert all_benchmarks[0].fn(**all_benchmarks[0].params) == 1
    assert has_expected_args(all_benchmarks[1].fn, {"param": 2})
    assert all_benchmarks[1].fn(**all_benchmarks[1].params) == 2


def test_product():
    @product(iter1=[1, 2], iter2=["a", "b"])
    def product_benchmark(iter1: int, iter2: str) -> tuple[int, str]:
        return iter1, iter2

    all_benchmarks = list(product_benchmark)
    assert len(all_benchmarks) == 4
    assert has_expected_args(all_benchmarks[0].fn, {"iter1": 1, "iter2": "a"})
    assert all_benchmarks[0].fn(**all_benchmarks[0].params) == (1, "a")
    assert has_expected_args(all_benchmarks[1].fn, {"iter1": 1, "iter2": "b"})
    assert all_benchmarks[1].fn(**all_benchmarks[1].params) == (1, "b")
    assert has_expected_args(all_benchmarks[2].fn, {"iter1": 2, "iter2": "a"})
    assert all_benchmarks[2].fn(**all_benchmarks[2].params) == (2, "a")
    assert has_expected_args(all_benchmarks[3].fn, {"iter1": 2, "iter2": "b"})
    assert all_benchmarks[3].fn(**all_benchmarks[3].params) == (2, "b")
