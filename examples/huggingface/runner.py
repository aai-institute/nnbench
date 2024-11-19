import nnbench


def main() -> None:
    console_reporter = nnbench.ConsoleReporter()
    runner = nnbench.BenchmarkRunner()
    result = runner.run("benchmark.py", tags=("per-class",))
    console_reporter.display(result)


if __name__ == "__main__":
    main()
