# Quickstart

Welcome! This quickstart guide will convey the basics needed to use nnbench.
You will define a benchmark, initialize a runner and reporter, and execute the benchmark, obtaining the results in the console in tabular format.

## A short scikit-learn model benchmark 

In the following simple example, we put the training and benchmarking logic in the same file. For more complex workloads, we recommend structuring your code into multiple files to improve project organization, similarly to unit tests.
See the user guides (TODO: Add guides) at the bottom of this page for inspiration.

```python
from sklearn.datasets import load_iris
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split

data = load_iris()
X, y = data.data, data.target

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3)

model = RandomForestClassifier()
model.fit(X_train, y_train)
```

To benchmark your model, you encapsulate the benchmark code into a function and apply the `@benchmark` decorator. 
This marks the function for collection to our benchmark runner later.

```python
import nnbench
import numpy as np
from sklearn import base, metrics


@nnbench.benchmark()
def accuracy(model: base.BaseEstimator, X_test: np.ndarray, y_test: np.ndarray) -> float:
    y_pred = model.predict(X_test)
    accuracy = metrics.accuracy_score(y_test, y_pred)
    return accuracy
```

Now we can instantiate a benchmark runner to collect and run the accuracy benchmark.
Then, we report the resulting accuracy metric by printing it to the terminal in a table.

```python
from nnbench import runner


r = runner.BenchmarkRunner()

# To collect in the current file, pass "__main__" as module name.
result = r.run("__main__", params={"model": model, "X_test": X_test, "y_test": y_test})

r.report(to='console', result=result)
```
The resulting output might look like this:

```bash
python benchmarks.py  


name         value
--------  --------
accuracy  0.933333
```
