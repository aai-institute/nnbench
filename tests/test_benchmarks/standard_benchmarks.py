import nnbench


@nnbench.benchmark(tags=("standard", "runner-collect"))
def double(x: int) -> int:
    return x * 2


@nnbench.benchmark(tags=("standard",))
def triple(y: int) -> int:
    return y * 3


@nnbench.benchmark(tags=("standard",))
def prod(x: int, y: int) -> int:
    return x * y
