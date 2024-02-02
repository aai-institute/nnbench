# Collecting and running benchmarks

nnbench provides the `BenchmarkRunner` as a compact interface to collect and run benchmarks selectively.

##  The abstract `BenchmarkRunner`  class
Let's first instantiate and then walk through the base class.

```python
from nnbench import BenchmarkRunner

runner = BenchmarkRunner()
```

Use the `BenchmarkRunner.collect()` method to collect benchmarks from files or directories.  
Assume we have the following benchmark setup:
```python
# dir_a/bm1.py
import nnbench

@nnbench.benchmark
def dummy_benchmark(a: int) -> int:
    return a
```

```python
# dir_b/bm2.py
import nnbench

@nnbench.benchmark(tags=("tag",))
def another_benchmark(b: int) -> int:
    return b

@nnbench.benchmark
def yet_another_benchmark(c: int) -> int:
    return c
```

```python
# dir_b/bm3.py
import nnbench
@nnbench.benchmark(tags=("tag",))
def the_last_benchmark(d: int) -> int:
    return d
```

Now we can collect benchmarks from files:

```python
runner.collect('dir_a/bm1.py')
```
Or directories:

```python
runner.collect('dir_b')
```

This collection can happen iteratively. So, after executing the two collections our runner has all four benchmarks ready for execution.

To remove the collected benchmarks again, use the `BenchmarkRunner.clear()` method.
You can also supply tags to the runner to selectively collect only benchmarks with the appropriate tag.
For example, after clearing the runner again, you can collect all benchmarks with the `"tag"` tag as such:

```python
runner.collect('dir_b', tags=("tag",))
```

To run the benchmarks, call the `BenchmarkRunner.run()` method and supply the necessary parameters required by the collected benchmarks.

```python
runner.run("dir_b", params={"b": 1, "c": 2, "d": 3})
```
