import nnbench


@nnbench.benchmark(tags=("runner-collect",))
def bad_random_number_gen() -> int:
    return 1
