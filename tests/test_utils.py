import inspect

import pytest

from nnbench.util import ismodule, modulename


@pytest.mark.parametrize("name,expected", [("sys", True), ("yaml", True), ("pipapo", False)])
def test_ismodule(name: str, expected: bool) -> None:
    actual = ismodule(name)
    assert expected == actual


@pytest.mark.parametrize(
    "name,expected",
    [("sys", "sys"), ("__main__", "__main__"), ("src/my/module.py", "src.my.module")],
)
def test_modulename(name: str, expected: str) -> None:
    actual = modulename(name)
    assert expected == actual


def has_expected_args(fn, expected_args):
    signature = inspect.signature(fn)
    params = signature.parameters
    return all(param in params for param in expected_args)
