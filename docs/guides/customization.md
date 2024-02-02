# Defining setup/teardown tasks, context, and `nnbench.Parameters`

This page introduces some customization options for benchmark runs.
These options can be helpful for tasks surrounding benchmark state management, such as automatic setup and cleanup, contextualizing results with context values, and defining typed parameters with the `nnbench.Parameters` class.

## Defining setup and teardown tasks

For some benchmarks, it is important to set certain configuration values and prepare the execution environment before running.
To do this, you can pass a setup task to all of the nnbench decorators via the `setUp` keyword:

```python
import os

import nnbench


def set_envvar(**params):
    os.environ["MY_ENV"] = "MY_VALUE"
    

@nnbench.benchmark(setUp=set_envvar)
def prod(a: int, b: int) -> int:
    return a * b
```

Similarly, to revert the environment state back to its previous form (or clean up any created resources), you can supply a finalization task with the `tearDown` keyword:

```python
import os

import nnbench


def set_envvar(**params):
    os.environ["MY_ENV"] = "MY_VALUE"
    

def pop_envvar(**params):
    os.environ.pop("MY_ENV")


@nnbench.benchmark(setUp=set_envvar, tearDown=pop_envvar)
def prod(a: int, b: int) -> int:
    return a * b
```

Both the setup and teardown task must take the exact same set of parameters as the benchmark function. To simplify function declaration, it is easiest to use a variadic keyword-only interface, i.e. `setup(**kwargs)`, as shown.

!!! tip
    This facility works exactly the same for the `@nnbench.parametrize` and `@nnbench.product` decorators.
    There, the specified setup and teardown tasks are run once before or after each of the resulting benchmarks respectively.

## Enriching benchmark metadata with context values

It is often useful to log specific environment metadata in addition to the benchmark's target metrics.
Such metadata can give a clearer picture of how certain models perform on a given hardware, how model architectures compare in performance, and much more.
In `nnbench`, you can give additional metadata to your benchmarks as **context values**.

A _context value_ is defined here as a key-value pair where `key` is a string, and `value` is any valid JSON value holding the desired information.
As an example, the context value `{"cpuarch": "arm64"}` gives information about the CPU architecture of the host machine running the benchmark.

A _context provider_ is a function taking no arguments and returning a Python dictionary of context values. The following is a basic example of a context provider:

```python
import platform

def platinfo() -> dict[str, str]:
    """Returns CPU arch, system name (Windows/Linux/Darwin), and Python version."""
    return {
        "system": platform.system(),
        "cpuarch": platform.machine(),
        "python_version": platform.python_version(),
    }
```

To supply context to your benchmarks, you can give a sequence of context providers to `BenchmarkRunner.run()`:

```python
import nnbench

# uses the `platinfo` context provider from above to log platform metadata.
runner = nnbench.BenchmarkRunner()
result = runner.run(__name__, params={}, context=[platinfo])
```

## Being type safe by using `nnbench.Parameters`

Instead of specifying your benchmark's parameters by using a raw Python dictionary, you can define a custom subclass of `nnbench.Parameters`:

```python
import nnbench
from dataclasses import dataclass


@dataclass(frozen=True)
class MyParams(nnbench.Parameters):
    a: int
    b: int


@nnbench.benchmark
def prod(a: int, b: int) -> int:
    return a * b


params = MyParams(a=1, b=2)
runner = nnbench.BenchmarkRunner()
result = runner.run(__name__, params=params)
```

While this does not have a concrete advantage in terms of type safety over a raw dictionary (all inputs will be checked against the types expected from the benchmark interfaces), it guards against accidental modification of parameters breaking reproducibility.
