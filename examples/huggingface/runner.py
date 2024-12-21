import nnbench


def main() -> None:
    benchmarks = nnbench.collect("benchmark.py", tags=("per-class",))
    reporter = nnbench.ConsoleReporter()
    result = nnbench.run(benchmarks)
    reporter.display(result)


if __name__ == "__main__":
    main()
