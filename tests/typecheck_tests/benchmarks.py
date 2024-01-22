import nnbench


@nnbench.benchmark
def double(x: int) -> int:
    return x*2


@nnbench.benchmark
def triple(y: int) -> int:
    return y * 3


@nnbench.benchmark
def prod(x: int, y: int) -> int:
    return x * y
