import nnbench


@nnbench.benchmark(tags=("standard", "duplicate"))
def double(x: int) -> int:
    return x * 2


@nnbench.benchmark(tags=("standard",))
def triple(y: int) -> int:
    return y * 3


@nnbench.benchmark(tags=("standard",))
def prod(x: int, y: int) -> int:
    return x * y


@nnbench.benchmark(tags=("with_default",))
def add_two(a: int = 0) -> int:
    return a + 2


@nnbench.benchmark(tags=("duplicate",))
def triple_int(x: int) -> int:
    return x * 3


@nnbench.benchmark(tags=("duplicate",))
def triple_str(x: str) -> str:
    return x * 3


@nnbench.benchmark(tags=("untyped",))
def increment(value):
    return value + 1
