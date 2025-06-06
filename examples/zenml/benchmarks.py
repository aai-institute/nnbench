from pipeline import ArrayMapping
from sklearn.base import BaseEstimator
from sklearn.metrics import accuracy_score

import nnbench


@nnbench.benchmark
def accuracy(model: BaseEstimator, data: ArrayMapping) -> float:
    X_test, y_test = data["X_test"], data["y_test"]
    y_pred = model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    return accuracy
