import nnbench


@nnbench.benchmark
def double(x: int) -> int:
    return x * 2
