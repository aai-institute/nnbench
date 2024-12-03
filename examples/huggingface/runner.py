import nnbench


def main() -> None:
    runner = nnbench.BenchmarkRunner()
    reporter = nnbench.ConsoleReporter()
    result = runner.run("benchmark.py", tags=("per-class",))
    reporter.display(result)


if __name__ == "__main__":
    main()
