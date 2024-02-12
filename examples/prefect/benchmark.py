import pickle
import sys
import time

import numpy as np
from sklearn import base, metrics

import nnbench


@nnbench.benchmark(tags=("metric",))
def mae(model: base.BaseEstimator, X_test: np.ndarray, y_test: np.ndarray) -> float:
    y_pred = model.predict(X_test)
    return metrics.mean_absolute_error(y_true=y_test, y_pred=y_pred)


@nnbench.benchmark(tags=("metric",))
def mse(model: base.BaseEstimator, X_test: np.ndarray, y_test: np.ndarray) -> float:
    y_pred = model.predict(X_test)
    return metrics.mean_squared_error(y_true=y_test, y_pred=y_pred)


@nnbench.benchmark(name="Model size (bytes)", tags=("model-meta",))
def modelsize(model: base.BaseEstimator) -> int:
    model_bytes = pickle.dumps(model)
    return sys.getsizeof(model_bytes)


@nnbench.benchmark(name="Inference time (s)", tags=("model-meta",))
def inference_time(model: base.BaseEstimator, X: np.ndarray, n_iter: int = 100) -> float:
    start = time.perf_counter()
    for i in range(n_iter):
        _ = model.predict(X)
    end = time.perf_counter()
    return (end - start) / n_iter
