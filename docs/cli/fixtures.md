# Using fixtures to supply parameters to benchmarks

One of the main problems when running `nnbench` on the command line is how to supply parameters.
Default values for benchmarks are one solution, but that does not scale well, and requires frequent code changes when values change.

Instead, nnbench borrows a bit of pytest's fixture concept to source parameters from special marker files, named `conf.py` in reference to pytest's `conftest.py`.

## How to define fixture values for benchmarks

Suppose you have a benchmark defined in a single file, `metrics.py`:

```python
# metrics.py
import nnbench


@nnbench.benchmark
def accuracy(model, data):
    ...
```

To supply `model` and `data` to the benchmark, define both values as return values of similarly named functions in a `conf.py` file in the same directory.
The layout of your benchmark directory should look like this:

```commandline
📂 benchmarks
┣━━ conf.py
┣━━ metrics.py
┣━━ ...
```

Inside your `conf.py` file, you might define your values as shown below. Note that currently, all fixtures must be raw Python callables, and must match input values of benchmarks exactly.

```python
# benchmarks/conf.py
def model():
    return MyModel()


def data():
    return TestDataset.load("path/to/my/dataset")
```

Then, nnbench will discover and auto-use these values when running this benchmark from the command line:

```commandline
$ nnbench run benchmarks.py 
```

!!! Warning
    Benchmarks with default values for their arguments will unconditionally use those defaults over potential fixtures.
    That is, for a benchmark `def add(a: int, b: int = 1)`, only the named parameter `a` will be resolved.

## Fixtures with inputs

Like in pytest, fixtures can consume inputs. However, in nnbench, fixtures can consume other inputs by name only within the same module scope, i.e. members within the same `conf.py`.

```python
# conf.py

# Revisiting the above example, we could also write the following:
def path() -> str:
    return "path/to/my/dataset"


def data(path):
    return TestDataset.load(path)

# ... but not this, since `config` is not a member of the conf.py module:
def model(config):
    return MyModel.load(config)
```

!!! Warning
    nnbench fixtures cannot have cycles in them - two fixtures may never depend on each other.

## Hierarchical `conf.py` files

nnbench also supports sourcing fixtures from different levels in a directory hierarchy.
Suppose we have a benchmark directory layout like this:

```commandline
📂 benchmarks
┣━━ 📂 nested
┃   ┣━━ conf.py
┃   ┗━━ model.py
┣━━ base.py
┗━━ conf.py
```

Let's assume that the benchmarks in `nested/model.py` consume some fixture values specific to them, and reuse some top-level fixtures as well.

```python
# benchmarks/conf.py

def path() -> str:
    return "path/to/my/dataset"


def data(path: str):
    """Test dataset, to be reused by all benchmarks."""
    return TestDataset.load(path)

# -------------------------------
# benchmarks/nested/conf.py

def model():
    """Model, needed only by the nested benchmarks."""
    return MyModel.load()
```

If we have a benchmark in `benchmarks/nested/model.py` defined like this:

```python
# benchmarks/nested/model.py

def accuracy(model, data):
    ...
```

nnbench will source the `model` fixture from `benchmarks/nested/conf.py`, and fall back to the top-level `benchmarks/conf.py` to obtain `data`.

!!! Info
    Just like in pytest, nnbench collects the deepest-level fixture it finds for a given name, so if `benchmarks/nested/conf.py` also defined a `data` fixture, `accuracy` would use that instead.
