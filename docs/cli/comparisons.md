# Comparing different benchmark runs with `nnbench compare`

To compare benchmark results across different runs, you can use the `nnbench compare` subcommand of the nnbench CLI.

```commandline
$ nnbench compare -h
usage: nnbench compare [-h]
                       [--comparison-file <JSON>]
                       [-P <name>] [-C <name>]
                       [-E <name>]
                       results [results ...]

positional arguments:
  results               Results to compare. Can be given as local files or remote URIs.

options:
  -h, --help            show this help message and exit
  --comparison-file <JSON>
                        A file containing comparison functions to run on benchmarking metrics.
  -P, --include-parameter <name>
                        Names of input parameters to display in the comparison table.
  -C, --include-context <name>
                        Context values to display in the comparison table.
                        Use dotted syntax for nested context values.
  -E, --extra-column <name>
                        Additional result data to display in the comparison table.
```

## A quick example

Suppose you run the following set of benchmarks:

```python
import nnbench


@nnbench.benchmark
def add(a: int, b: int) -> int:
    return a + b


@nnbench.benchmark
def sub(a: int, b: int) -> int:
    return a + b
```

two times, with the parameters `a = 1, b = 2` and `a = 1, b = 3`, respectively.
We will now set up a comparison between the two runs by crafting a `comparisons.json` file that defines the comparisons we will apply on the respective metrics:

```json
# file: comparisons.json
{
    "add": {
        "class": "nnbench.compare.GreaterEqual"
    },
    "sub": {
        "class": "nnbench.compare.AbsDiffLessEqual",
        "kwargs": {
            "thresh": 0.5
        }
    }
}
```

The structure itself is simple: The contents map the metric names (`"add"` and `"sub"` in this case) to their respective comparison classes.

Each comparison is encoded by the class name, given as a fully qualified Python module path that can be imported via `importlib`, and a `kwargs` dictionary, which will be passed to the chosen class on construction (i.e. to its `__init__()`) method.

!!! Note
    You are responsible for only passing keyword arguments that are expected by the chosen comparison class.


Now, running `nnbench compare <records> --comparison-file=comparisons.json` will produce a table like the following (with an sqlite database holding the two results as an example):

```commandline
$ nnbench compare sqlite://hello.db --comparison-file=comp.json      
Comparison strategy: All vs. first
Comparisons:
    add: x ≥ y
    sub: |x - y| <= 0.50
┏━━━━━━━━━━━━━━━━━━━━━━━━━━━━━┳━━━━━━━━━━━━━━━━━┳━━━━━━━━━━━━━━━━━┳━━━━━━━━┓
┃ Run Name                    ┃ add             ┃ sub             ┃ Status ┃
┡━━━━━━━━━━━━━━━━━━━━━━━━━━━━━╇━━━━━━━━━━━━━━━━━╇━━━━━━━━━━━━━━━━━╇━━━━━━━━┩
│ nnbench-1749042461424137000 │ 3.00            │ -1.0            │        │
│ nnbench-1749042477901108000 │ 4.00 (vs. 3.00) │ -2.0 (vs. -1.0) │ ✅❌    │
└─────────────────────────────┴─────────────────┴─────────────────┴────────┘
```

The comparison strategy is printed first, listed as "All vs. first" (the only strategy currently supported), meaning that all results are compared against the result of the first given run.
After that, the comparisons are listed as concise human-readable mathematical expressions.

In this case, we want to ensure that the candidate results have a greater value for "add" (indicated by the "x ≥ y" formula, where x is the candidate value and y is the value in the first record), and a value for sub that is within `0.5` compared to the first result.

As the value for `add` is greater in the second run, this comparison succeeded, as indicated by the green checkmark, but the `sub` comparison failed, since both results differ by `1.0`, which is more than the allowed `0.5`.

## Extending comparisons

To craft your own comparison, subclass the `nnbench.compare.AbstractComparison` class.
You can also define your own comparison functions, which need to implement the following protocol:

```python
class Comparator(Protocol):
    def __call__(self, val1: Any, val2: Any) -> bool: ...
```

!!! Note
    One-versus-all metric comparisons and running user-defined comparisons are planned for a future release of `nnbench`.
