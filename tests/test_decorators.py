import pytest

import nnbench
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

    assert len(parametrized_benchmark) == 2
    assert has_expected_args(parametrized_benchmark[0].fn, {"param": 1})
    assert parametrized_benchmark[0].fn() == 1
    assert has_expected_args(parametrized_benchmark[1].fn, {"param": 2})
    assert parametrized_benchmark[1].fn() == 2


def test_parametrize_with_duplicate_parameters():
    with pytest.warns(UserWarning, match="duplicate"):

        @parametrize([{"param": 1}, {"param": 1}])
        def parametrized_benchmark(param: int) -> int:
            return param


def test_product():
    @product(iter1=[1, 2], iter2=["a", "b"])
    def product_benchmark(iter1: int, iter2: str) -> tuple[int, str]:
        return iter1, iter2

    assert len(product_benchmark) == 4
    assert has_expected_args(product_benchmark[0].fn, {"iter1": 1, "iter2": "a"})
    assert product_benchmark[0].fn() == (1, "a")
    assert has_expected_args(product_benchmark[1].fn, {"iter1": 1, "iter2": "b"})
    assert product_benchmark[1].fn() == (1, "b")
    assert has_expected_args(product_benchmark[2].fn, {"iter1": 2, "iter2": "a"})
    assert product_benchmark[2].fn() == (2, "a")
    assert has_expected_args(product_benchmark[3].fn, {"iter1": 2, "iter2": "b"})
    assert product_benchmark[3].fn() == (2, "b")


def test_product_with_duplicate_parameters():
    with pytest.warns(UserWarning, match="duplicate"):

        @product(iter=[1, 1])
        def product_benchmark(iter: int) -> int:
            return iter
