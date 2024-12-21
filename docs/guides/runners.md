# Collecting and running benchmarks

nnbench provides the `nnbench.collect` and `nnbench.run` APIs as a compact interface to collect and run benchmarks selectively.

Use the `nnbench.collect()` method to collect benchmarks from files or directories.  
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
import nnbench


benchmarks = nnbench.collect('dir_a/bm1.py')
```
Or directories:

```python
benchmarks = nnbench.collect('dir_b')
```

You can also supply tags to the runner to selectively collect only benchmarks with the appropriate tag.
For example, after clearing the runner again, you can collect all benchmarks with the `"tag"` tag as such:

```python
import nnbench


tagged_benchmarks = nnbench.collect('dir_b', tags=("tag",))
```

To run the benchmarks, call the `nnbench.run()` method and supply the necessary parameters required by the collected benchmarks.

```python
result = nnbench.run(benchmarks, params={"b": 1, "c": 2, "d": 3})
```
