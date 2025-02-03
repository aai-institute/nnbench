import time

import nnbench


@nnbench.benchmark
def add(a: int, b: int) -> int:
    time.sleep(10)
    return a + b
