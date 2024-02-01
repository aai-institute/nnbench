# Defining benchmarks with decorators

To benchmark your machine learning code in nnbench, define your key metrics in Python functions and apply one of the provided decorators.
The available decorators are 
- `@nnbench.benchmark`, which runs a benchmark with supplied parameters,
- `@nnbench.parametrize`, which runs several benchmarks with the supplied parameter configurations,
- `@nnbench.product`, which runs benchmarks with all parameter combinations that arise from the supplied values. 

First we introduce a small machine learning example which we will subsequently use to motivate the use of the three benchmark decorators.

We recommend to split the model training, benchmark definition, and benchmark running into different files. In this guide, these are called `training.py`, `benchmarks.py`, and `main.py`.

## Example
Let us consider an example where we want to evaluate a `scikit-learn` random forest classifier on the Iris dataset.
For this purpose, we will define several helper functions inside a file, `training.py`. We use `prepare_data()`, to load the dataset,  `train_rf()` to train a random forest model with the specified parameters, and `accuracy()` to calculate the accuracy of the supplied model on the given dataset.

```python
# training.py
import numpy as np
from sklearn import base, metrics
from sklearn.datasets import load_iris
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split


def prepare_data() -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    data = load_iris()
    X, y = data.data, data.target
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3)
    return X_train, X_test, y_train, y_test


def train_rf(X_train: np.ndarray, y_train: np.ndarray, n_estimators: int, max_depth: int, random_state: int = 42) -> RandomForestClassifier:
    model = RandomForestClassifier(n_estimators=n_estimators, max_depth=max_depth, random_state=random_state)
    model.fit(X_train, y_train)
    return model


def accuracy(model: base.BaseEstimator, y_test: np.ndarray, y_pred: np.ndarray) -> float:
    accuracy = metrics.accuracy_score(y_test, y_pred)
    return accuracy
```

## `@benchmark` for single benchmarks
Now, we define our benchmarks in a new file called `benchmarks.py`.
We first encapsulate the benchmark logic into a function, `benchmark_accuracy()` which prepares the data, trains a classifier, and lastly, obtains the accuracy.
To mark such a function as a benchmark, we apply the `@benchmark` decorator.

```python
# benchmarks.py
import nnbench
from training import prepare_data, train_rf, accuracy

@nnbench.benchmark()
def benchmark_accuracy(n_estimators: int, max_depth: int, random_state: int) -> float:
    X_train, X_test, y_train, y_test = prepare_data()
    rf = train_rf(X_train=X_train, y_train=y_train, n_estimators=n_estimators,
                  max_depth=max_depth, random_state=random_state)
    acc = accuracy(model=rf, X_test=X_test, y_test=y_test)
    return acc
```

!!! warning
    This training benchmark is designed as a local, simple, and self-contained example to showcase nnbench. 
    In a real world scenario, to follow best practices, you may want to separate the data preparation and model training steps from the benchmarking logic and pass the corresponding artifacts as a parameter to the benchmark. See the user guide for more information.

Lastly, we set up a benchmark runner in the `main.py`. There, we supply the parameters (`n_estimators`, `max_depth`, `random_state`) necessary in the function definition as a dictionary to the `params` keyword argument. 

```python
# main.py
from nnbench import runner

r = runner.BenchmarkRunner()
result = r.run("./benchmarks.py", params={"n_estimators": 100, "max_depth": 5, "random_state": 42})
r.report(to='console', result=result)
```

When we execute the `main.py` we get the following output:


```bash
python main.py  


name         value
--------  --------
accuracy  0.933333
```

## `@nnbench.parametrize` for multiple configuration benchmarks

Sometimes, we are not only interested in the performance of a model for given parameters but want to compare the performance for different configurations. 
To achieve this, we can turn our single accuracy benchmark in the `benchmarks.py` file into a parametrized benchmark.
To do this, replace the decorator with `@nnbench.parametrize` and supply the parameter combinations of choice as dictionaries in the first argument.

```python
# benchmarks.py
import nnbench
from training import prepare_data, train_rf, accuracy

@nnbench.parametrize(
    ({"n_estimators": 10, "max_depth": 2},
    {"n_estimators": 50, "max_depth": 5},
    {"n_estimators": 100, "max_depth": 10})
)
def benchmark_accuracy(n_estimators: int, max_depth: int, random_state: int) -> float:
    X_train, X_test, y_train, y_test = prepare_data()
    rf = train_rf(X_train=X_train, y_train=y_train, n_estimators=n_estimators,
                  max_depth=max_depth, random_state=random_state)
    acc = accuracy(model=rf, X_test=X_test, y_test=y_test)
    return acc
```

Notice that the parametrization is still incomplete, as we did not supply a `random_state` argument.
The unfilled arguments are given in `BenchmarkRunner.run()` via a dictionary passed as the `params` keyword argument.

```python
# main.py
from nnbench import runner

r = runner.BenchmarkRunner()
result = r.run("./benchmarks.py", params={"random_state": 42})
r.report(to='console', result=result)
```

Executing the parametrized benchmark, we get an output similar to this:

```bash
python main.py  


name                                                 value
------------------------------------------------  --------
benchmark_accuracy_n_estimators=10_max_depth=2    0.955556
benchmark_accuracy_n_estimators=50_max_depth=5    0.866667
benchmark_accuracy_n_estimators=100_max_depth=10  0.911111
```

## `@nnbench.product` for benchmarks over parameter configuration grids

In case we want to run a benchmark scan for all possible combinations of a set of parameters, we can use the `@nnbench.product` decorator to supply the different values for each parameter.

```python
# benchmarks.py
import nnbench
from training import prepare_data, train_rf, accuracy

@nnbench.product(n_estimators=[10, 50, 100], max_depth=[2, 5, 10])
def benchmark_accuracy_product(n_estimators: int, max_depth: int, random_state: int) -> float:
    X_train, X_test, y_train, y_test = prepare_data()
    rf = train_rf(X_train=X_train, y_train=y_train, n_estimators=n_estimators,
                  max_depth=max_depth, random_state=random_state)
    acc = accuracy(model=rf, X_test=X_test, y_test=y_test)
    return acc
```

We still provide the `random_state` parameter to the runner directly, like we did with the `@nnbench.parametrize` decorator.
By executing the benchmark, we get results for all combinations of `n_estimators` and `max_depth`.
It looks similar to this:

```bash
python main.py  


name                                                 value
------------------------------------------------  --------
benchmark_accuracy_n_estimators=10_max_depth=2    0.933333
benchmark_accuracy_n_estimators=10_max_depth=5    0.955556
benchmark_accuracy_n_estimators=10_max_depth=10   0.977778
benchmark_accuracy_n_estimators=50_max_depth=2    0.933333
benchmark_accuracy_n_estimators=50_max_depth=5    0.911111
benchmark_accuracy_n_estimators=50_max_depth=10   0.977778
benchmark_accuracy_n_estimators=100_max_depth=2   0.933333
benchmark_accuracy_n_estimators=100_max_depth=5   0.955556
benchmark_accuracy_n_estimators=100_max_depth=10  0.955556
```
