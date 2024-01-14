from mlbench import benchmark


@benchmark
def double(x: int) -> int:
    return x * 2
