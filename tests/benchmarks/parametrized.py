import nnbench


@nnbench.parametrize([{"a": 1}, {"a": 2}], tags=("parametrized",))
def double(a: int) -> int:
    return 2 * 2
