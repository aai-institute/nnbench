import time

from tests.cli import DELAY_SECONDS

import nnbench


@nnbench.benchmark
def mul(a: int, b: int) -> int:
    time.sleep(DELAY_SECONDS)
    return a * b
