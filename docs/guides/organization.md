# How to efficiently organize benchmark code

To efficiently organize benchmarks and keeping your setup modular, you can follow a few guidelines.

## Tip 1: Separate benchmarks from project code

This tip is well known from other software development practices such as unit testing.
To improve project organization, consider splitting off your benchmarks into their own modules or even directories, if you have multiple benchmark workloads.

An example project layout can look like this, with benchmarks as a separate directory at the top-level:

```
my-project/
├── benchmarks/ # <- contains all benchmarking Python files.
├── docs/
├── src/
├── .pre-commit-config.yaml
├── pyproject.toml
├── README.md
└── ...
```

This keeps the benchmarks neatly grouped together while siloing them away from the actual project code.
Since you will most likely not run your benchmarks in a production setting, this is also advantageous for packaging, as the `benchmarks/` directory does not ship by default in this configuration.

## Tip 2: Group benchmarks by common attributes

To maintain good organization within your benchmark directory, you can group similar benchmarks into their own Python files.
As an example, if you have a set of benchmarks to establish data quality, and benchmarks for scoring trained models on curated data, you could structure them as follows:

```
benchmarks/
├── data_quality.py
├── model_perf.py
└── ...
```

This is helpful when running multiple benchmark workloads separately, as you can just point your benchmark runner to each of these separate files:

```python
from nnbench.runner import BenchmarkRunner

runner = BenchmarkRunner()
data_metrics = runner.run("benchmarks/data_quality.py", params=...)
# same for model metrics, where instead you pass benchmarks/model_perf.py.
model_metrics = runner.run("benchmarks/model_perf.py", params=...)
```

## Tip 3: Attach tags to benchmarks for selective filtering

For structuring benchmarks within files, you can also use **tags**, which are tuples of strings attached to a benchmark:

```python
# benchmarks/data_quality.py
import nnbench


@nnbench.benchmark(tags=("foo",))
def foo1(data) -> float:
    ...


@nnbench.benchmark(tags=("foo",))
def foo2(data) -> int:
    ...


@nnbench.benchmark(tags=("bar",))
def bar(data) -> int:
    ...
```

Now, to only run data quality benchmarks marked "foo", pass the corresponding tag to `BenchmarkRunner.run()`:

```python
from nnbench.runner import BenchmarkRunner

runner = BenchmarkRunner()
foo_data_metrics = runner.run("benchmarks/data_quality.py", params=..., tags=("foo",))
```

!!!tip
    This concept works exactly the same when creating benchmarks with the `@nnbench.parametrize` and `@nnbench.product` decorators.
