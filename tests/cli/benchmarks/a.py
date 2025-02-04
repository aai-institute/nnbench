import time

from tests.cli import DELAY_SECONDS

import nnbench


@nnbench.benchmark
def add(a: int, b: int) -> int:
    time.sleep(DELAY_SECONDS)
    return a + b
