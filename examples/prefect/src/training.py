from __future__ import annotations

import numpy as np
from prefect import flow, task
from sklearn import base
from sklearn.datasets import make_regression
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split


@task
def make_regression_data(
    random_state: int, n_samples: int = 100, n_features: int = 1, noise: float = 0.2
) -> tuple[np.ndarray, np.ndarray]:
    X, y = make_regression(
        n_samples=n_samples, n_features=n_features, noise=noise, random_state=random_state
    )
    return X, y


@task
def make_train_test_split(
    X: np.ndarray, y: np.ndarray, random_state: int, test_size: float = 0.2
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state
    )
    return X_train, y_train, X_test, y_test


@task
def train_linear_regression(X: np.ndarray, y: np.ndarray) -> base.BaseEstimator:
    model = LinearRegression()
    model.fit(X, y)
    return model


@flow
def prepare_regression_data(
    random_state: int = 42, n_samples: int = 100, n_features: int = 1, noise: float = 0.2
) -> tuple[np.ndarray, ...]:
    X, y = make_regression_data(
        random_state=random_state, n_samples=n_samples, n_features=n_features, noise=noise
    )
    X_train, y_train, X_test, y_test = make_train_test_split(X=X, y=y, random_state=random_state)
    return X_train, y_train, X_test, y_test


@flow
async def prepare_regressor_and_test_data(
    data_params: dict[str, int | float] | None = None,
) -> tuple[base.BaseEstimator, np.ndarray, np.ndarray]:
    if data_params is None:
        data_params = {}
    X_train, y_train, X_test, y_test = prepare_regression_data(**data_params)
    model = train_linear_regression(X=X_train, y=y_train)
    return model, X_test, y_test
