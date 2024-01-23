import nnbench


@nnbench.benchmark
def double(x: int) -> int:
    return x * 2


@nnbench.benchmark
def triple_str(x: str) -> str:
    return x * 3
