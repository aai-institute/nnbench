from mlbench.benchmark import Benchmark


def double(x: int) -> int:
    return x * 2


a = Benchmark(double)
