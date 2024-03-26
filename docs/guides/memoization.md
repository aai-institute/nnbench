# Using memoization for memory efficient benchmarks

In machine learning workloads, models and datasets of greater-than-memory size are frequently encountered.
Especially when loading and benchmarking several models in succession, for example with a parametrization, available memory can quickly become a bottleneck.

To address this problem, this guide introduces **memos** as a way to reduce memory pressure when benchmarking multiple memory-intensive models and datasets sequentially.

## Using the `nnbench.Memo` class
// TODO: Move the class up into the main export.

The key to efficient memory utilization in nnbench is _memoization_.
nnbench itself provides the `nnbench.Memo` class, a generic base class that can be subclassed to yield a value and cache it for subsequent invocations.

To subclass a memo, overload the `Memo.__call__()` operator like so:

```python
import numpy as np
import nnbench

class MyType:
    """Contains a huge array, similarly to a model."""
    a: np.ndarray = np.zeros((10000, 10000))

class MyMemo(nnbench.Memo[MyType]):
    
    # TODO: Add a memo cache decorator.
    def __call__(self) -> MyType:
        return MyType()
```

`nnbench.Memo` objects do not take any arguments, meaning that all external state necessary to compute the value needs to be passed in the `Memo.__init__()` function.
In this way, nnbench's memos work similarly to e.g. [React's useMemo hook](https://react.dev/reference/react/useMemo).

!!! Warning
    You must explicitly hint the returned type in the `Memo.__call__()` annotation, which needs to match the generic type specialization (the type in the square brackets in the class definition),
    otherwise nnbench will throw errors when validating benchmark parameters.

## Supplying memoized values to benchmarks

Memoization is especially useful when parametrizing benchmarks over models and datasets.

Suppose we have a `Model` class wrapping a large (in the order of available memory) NumPy array.
If we cannot load all models into memory at the same time in a benchmark run, we can load the (serialized) models one by one using nnbench memos.

```python
import nnbench

import numpy as np


class Model:
    def __init__(self, arr: np.ndarray):
        self.array = arr
    
    def apply(self, arr: np.ndarray) -> np.ndarray:
        return self.array @ arr

class ModelMemo(nnbench.Memo[Model]):
    def __init__(self, path):
        self.path = path
    
    # TODO: Add a memo cache decorator.
    def __call__(self) -> Model:
        arr = np.load(self.path)
        return Model(arr)


@nnbench.product(
    model=[ModelMemo(p) for p in ("model1.npz", "model2.npz", "model3.npz")]
)
def accuracy(model: Model, data: np.ndarray) -> float:
    return np.sum(model.apply(data))
```

//TODO: Add tearDown task clearing the memo cache.

After each benchmark, each model memo's corresponding value is evicted from nnbench's memoization cache.

!!! Warning
    If you evict a value before its last use in a benchmark, it will be recomputed, potentially slowing down benchmark execution by a lot.

## Summary

- Use `nnbench.Memo`s to lazy-load and explicitly control the lifetime of objects with large memory footprint.
- Annotate memos with their specialized type to avoid problems with nnbench's type checking and parameter validation. 
- Use teardown tasks after benchmarks to evict memoized values from the memo cache.
