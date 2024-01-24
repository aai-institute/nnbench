import nnbench


@nnbench.benchmark
def increment(value):
    return value + 1
