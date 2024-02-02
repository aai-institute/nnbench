# Running benchmarks with the benchmark runner

nnbench provides the `BenchmarkRunner` as a compact interface to collect and run benchmarks selectively.
You can either use it directly or instantiate a derivative class to override selected methods and implement you own logic.


##  The abstract `BenchmarkRunner`  class
Let's first instantiate and then walk through the abstract class.

```python
from nnbench import BenchmarkRunner

runner = BenchmarkRunner()
```

Now we can use the `BenchmarkRunner.collect` method to collect benchmarks from files or directories.
Lets assume we have the following benchmark setup:
```python
# ./performance/model_benchmark.py
import nnbench
import numpy as np
from sklearn import base, metrics

@nnbench.benchmark
def accuracy(model: base.BaseEstimator, X_test: np.ndarray, y_test: np.ndarray) -> float:
    y_pred = model.predict(X_test)
    accuracy = metrics.accuracy_score(y_test, y_pred)
    return accuracy
```

```python
# ./model_info/inference_resources.py
import nnbench
import numpy as np
import time
from sklearn import base
from memory_profiler import memory_usage

@nnbench.benchmark(tags=("inference-resources",))
def avg_inference_time(model: base.BaseEstimator, X_test: np.ndarray, steps: int) -> float:
    total_time = 0.0

    for _ in range(steps):
        start_time = time.time()
        model.predict(X_test)
        end_time = time.time()
        total_time += end_time - start_time

    average_time = total_time / steps
    return average_time

@nnbench.benchmark(tags=("inference-resources",))
def memory_usage_during_inference(model: base.BaseEstimator, X_test: np.ndarray) -> float:
    def predict():
        model.predict(X_test)

    mem_usage = memory_usage(predict, max_usage=True)
    return mem_usage
```

```python
# ./model_info/storage_resources.py
import nnbench
import numpy as np
from sklearn import base

@nnbench.benchmark
def model_size_mb(model: base.BaseEstimator) -> float:
    total_size_bytes = 0

    for attr, value in model.__dict__.items():
        if isinstance(value, np.ndarray):
            total_size_bytes += value.nbytes
        elif isinstance(value, list) or isinstance(value, dict):
            for item in value:
                if isinstance(item, np.ndarray):
                    total_size_bytes += item.nbytes

    size_mb = total_size_bytes / (1024 * 1024)
    return size_mb
```

Now we can collect benchmarks from files:

```python
runner.collect('./performance/model_benchmark.py')
```
Or directories:

```python
runner.collect('./model_info')
```

This collection can happen iteratively. So, after executing the two collections our runner has all four benchmarks ready for execution.
If we want to remove the collected benchmarks, we can use the `BenchmarkRunner.clear` method.
We can also supply tags to the collection. Then the runner will only collect benchmarks with the appropriate tag.
Hence, we can - after clearing the runner - collect all benchmarks with the "inference-resources" tag:

```python
runner.collect('./model_info', tags=("inference-resources",))
```
Note the trailing comma which is necessary for single entires in parenthesis to be interpreted as a tuple, the proper datatypes for tags.

We can execute our benchmark with the `BenchmarkRunner.run` method. However, we have to pass all the necessary parameters for our benchmark as well. 
If we can import a scikit-learn `model` from another file, let's say `traininf.py`, the code to execute the "inference-resources" benchmarks looks like this: 

# TODO: currently this does not work. We have not added a path to the runner. Should it work without?
```python
from training import model, X_test

runner.run({"model": model, "X_test": X_test, "steps": 25})
```

## Custom benchmark runners
