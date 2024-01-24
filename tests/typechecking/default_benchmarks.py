import nnbench


@nnbench.benchmark
def add_two(a: int = 0) -> int:
    return a + 2
