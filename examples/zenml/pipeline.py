# /// script
# requires-python = ">=3.10"
# dependencies = [
#     "scikit-learn",
#     "zenml",
#     "nnbench",
# ]
# ///

"""
The scikit-learn random forest example with nnbench, as a ZenML pipeline.

Demonstrates the use of nnbench to collect and log metrics right after training.
"""

from dataclasses import dataclass
from pathlib import Path

import numpy as np
from sklearn.datasets import load_iris
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from zenml import log_metadata, pipeline, step

import nnbench

HERE = Path(__file__).parent

ArrayMapping = dict[str, np.ndarray]


MAX_DEPTH = 5
N_ESTIMATORS = 100
RANDOM_STATE = 42


@dataclass(frozen=True)
class MNISTTestParameters(nnbench.Parameters):
    model: RandomForestClassifier
    data: ArrayMapping


@step
def load_iris_dataset() -> ArrayMapping:
    """
    Load and split Iris data from scikit-learn.

    Returns
    -------
    ArrayMapping
        Iris dataset as NumPy arrays, split into training and test data.
    """

    data = load_iris()
    X, y = data.data, data.target
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3)

    iris = {
        "X_train": X_train,
        "X_test": X_test,
        "y_train": y_train,
        "y_test": y_test,
    }

    return iris


@step
def train_model(
    data: ArrayMapping,
    n_estimators: int = N_ESTIMATORS,
    max_depth: int = MAX_DEPTH,
    random_state: int = RANDOM_STATE,
) -> RandomForestClassifier:
    model = RandomForestClassifier(
        n_estimators=n_estimators, max_depth=max_depth, random_state=random_state
    )
    X_train, y_train = data["X_train"], data["y_train"]
    model.fit(X_train, y_train)
    return model


@step
def benchmark_model(model: RandomForestClassifier, data: ArrayMapping) -> None:
    """Evaluate the model and log metrics in nnbench."""

    # the nnbench portion.
    benchmarks = nnbench.collect(HERE)
    params = MNISTTestParameters(model=model, data=data)
    result = nnbench.run(benchmarks, name="nnbench-iris-run", params=params)

    # Log metrics to the step.
    log_metadata(metadata=result.to_json())


@pipeline
def mnist_jax():
    """Load Iris data, train, and benchmark a simple random forest model."""
    iris = load_iris_dataset()
    model = train_model(iris)
    benchmark_model(model, iris)


if __name__ == "__main__":
    mnist_jax()
