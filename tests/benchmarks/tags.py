import nnbench


@nnbench.benchmark(tags=("tag1",))
def subtract(a: int, b: int) -> int:
    return a - b


@nnbench.benchmark(tags=("tag1", "tag2"))
def decrement(a: int) -> int:
    return a - 1


@nnbench.benchmark()
def identity(x: int) -> int:
    return x
