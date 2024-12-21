# nnbench: A small framework for benchmarking machine learning models

Welcome to nnbench, a framework for benchmarking machine learning models.
The main goals of this project are

1. To provide a portable, easy-to-use solution for model evaluation that leads to better ML experiment organization, and
2. To integrate with experiment and metadata tracking solutions for easy adoption.

On a high level, you can think of nnbench as "pytest for ML models" - you define benchmarks similarly to test cases, collect them, and selectively run them based on model type, markers, and environment info.

What's new is that upon completion, you can stream the resulting data to any sink of your choice (including multiple at the same), which allows easy integration with experiment trackers and metadata stores.

See the [quickstart](https://aai-institute.github.io/nnbench/latest/quickstart/) for a lightning-quick demo, or the [examples](https://aai-institute.github.io/nnbench/latest/tutorials/) for more advanced usages.

## Installation

⚠️ nnbench is an experimental project - expect bugs and sharp edges.

Install it directly from source, for example either using `pip` or `uv`:

```shell
pip install nnbench
# or
uv add nnbench
```

## A ⚡️- quick demo

To understand how nnbench works, you can run the following in your Python interpreter:

```python
# example.py
import nnbench


@nnbench.benchmark
def product(a: int, b: int) -> int:
    return a * b


@nnbench.benchmark
def power(a: int, b: int) -> int:
    return a ** b


reporter = nnbench.ConsoleReporter()
# first, collect the above benchmarks directly from the current module...
benchmarks = nnbench.collect("__main__")
# ... then run the benchmarks with the parameters `a=2, b=10`...
record = nnbench.run(benchmarks, params={"a": 2, "b": 10})
reporter.display(record)  # ...and print the results to the terminal.

# results in a table look like the following:
# ┏━━━━━━━━━━━┳━━━━━━━┳━━━━━━━━━━━━━━━━┳━━━━━━━━━━━━━━━━━━━┓
# ┃ Benchmark ┃ Value ┃ Wall time (ns) ┃ Parameters        ┃
# ┡━━━━━━━━━━━╇━━━━━━━╇━━━━━━━━━━━━━━━━╇━━━━━━━━━━━━━━━━━━━┩
# │ product   │ 20    │ 1917           │ {'a': 2, 'b': 10} │
# │ power     │ 1024  │ 583            │ {'a': 2, 'b': 10} │
# └───────────┴───────┴────────────────┴───────────────────┘
```
Watch the following video for a high level overview of the capabilities and inner workings of nnbench.

[![nnbench overview video thumbnail](https://img.youtube.com/vi/CT9bKq-U8ZQ/0.jpg)](https://www.youtube.com/watch?v=CT9bKq-U8ZQ)

For a more realistic example of how to evaluate a trained model with a benchmark suite, check the [Quickstart](https://aai-institute.github.io/nnbench/latest/quickstart/).
For even more advanced usages of the library, you can check out the [Examples](https://aai-institute.github.io/nnbench/latest/tutorials/) in the documentation.

## Contributing

We encourage and welcome contributions from the community to enhance the project.
Please check [discussions](https://github.com/aai-institute/nnbench/discussions) or raise an [issue](https://github.com/aai-institute/nnbench/issues) on GitHub for any problems you encounter with the library.

For information on the general development workflow, see the [contribution guide](CONTRIBUTING.md).

## License

The nnbench library is distributed under the [Apache-2 license](LICENSE).
