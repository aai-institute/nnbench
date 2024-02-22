import nnbench


@nnbench.benchmark
def prod(a: int, b: int) -> int:
    return a * b


@nnbench.benchmark
def sum(a: int, b: int) -> int:
    return a + b
