# /// script
# requires-python = ">=3.11"
# dependencies = [
#     "jax[cpu]",
#     "optax",
#     "flax",
#     "zenml",
#     "nnbench @ /Users/nicholasjunge/Workspaces/python/nnbench",
#     "requests",
#     "aiohttp",
# ]
# ///

"""
The JAX MNIST example with nnbench, as a ZenML pipeline.

Demonstrates the use of nnbench to collect and log metrics right after training.

Source: https://github.com/google/flax/blob/main/examples/mnist
"""

import random
from collections.abc import Mapping
from dataclasses import dataclass
from functools import partial
from pathlib import Path

import fsspec
import jax
import jax.numpy as jnp
import jax.random as jr
import numpy as np
import optax
from flax import nnx  # The Flax NNX API.
from flax.training.train_state import TrainState
from zenml import log_metadata, pipeline, step

import nnbench

HERE = Path(__file__).parent

ArrayMapping = dict[str, jax.Array | np.ndarray]

INPUT_SHAPE = (28, 28, 1)  # H x W x C (= 1, BW grayscale images)
NUM_CLASSES = 10
BATCH_SIZE = 128
EPOCHS = 1
LEARNING_RATE = 0.1
MOMENTUM = 0.9


@dataclass(frozen=True)
class MNISTTestParameters(nnbench.Parameters):
    params: Mapping[str, jax.Array]
    data: ArrayMapping

class CNN(nnx.Module):
  """A simple CNN model."""

  def __init__(self, *, rngs: nnx.Rngs):
    self.conv1 = nnx.Conv(1, 32, kernel_size=(3, 3), rngs=rngs)
    self.conv2 = nnx.Conv(32, 64, kernel_size=(3, 3), rngs=rngs)
    self.avg_pool = partial(nnx.avg_pool, window_shape=(2, 2), strides=(2, 2))
    self.linear1 = nnx.Linear(3136, 256, rngs=rngs)
    self.linear2 = nnx.Linear(256, 10, rngs=rngs)

  def __call__(self, x):
    x = self.avg_pool(nnx.relu(self.conv1(x)))
    x = self.avg_pool(nnx.relu(self.conv2(x)))
    x = x.reshape(x.shape[0], -1)  # flatten
    x = nnx.relu(self.linear1(x))
    x = self.linear2(x)
    return x


def loss_fn(model: CNN, batch):
  logits = model(batch['image'])
  loss = optax.softmax_cross_entropy_with_integer_labels(
    logits=logits, labels=batch['label']
  ).mean()
  return loss, logits


@nnx.jit
def train_step(model: CNN, optimizer: nnx.Optimizer, metrics: nnx.MultiMetric, batch):
  """Train for a single step."""
  grad_fn = nnx.value_and_grad(loss_fn, has_aux=True)
  (loss, logits), grads = grad_fn(model, batch)
  metrics.update(loss=loss, logits=logits, labels=batch['label'])  # In-place updates.
  optimizer.update(grads)  # In-place updates.


@nnx.jit
def eval_step(model: CNN, metrics: nnx.MultiMetric, batch):
  loss, logits = loss_fn(model, batch)
  metrics.update(loss=loss, logits=logits, labels=batch['label'])  # In-place updates.


@step
def load_mnist() -> ArrayMapping:
    """
    Load MNIST dataset using fsspec.

    Returns
    -------
    ArrayMapping
        Versioned dataset as numpy arrays, split into training and test data.
    """

    if Path(HERE / "mnist.npz").exists():
        data = np.load(HERE / "mnist.npz")
        return dict(data)

    mnist: ArrayMapping = {}

    baseurl = "https://storage.googleapis.com/cvdf-datasets/mnist/"

    for key, file in [
        ("x_train", "train-images-idx3-ubyte.gz"),
        ("x_test", "t10k-images-idx3-ubyte.gz"),
        ("y_train", "train-labels-idx1-ubyte.gz"),
        ("y_test", "t10k-labels-idx1-ubyte.gz"),
    ]:
        with fsspec.open(baseurl + file, compression="gzip") as f:
            if key.startswith("x"):
                mnist[key] = np.frombuffer(f.read(), np.uint8, offset=16).reshape((-1, 28, 28))
            else:
                mnist[key] = np.frombuffer(f.read(), np.uint8, offset=8)

    # save the data locally after download.
    np.savez_compressed(HERE / "mnist.npz", **mnist)

    return mnist


@step
def train_epoch(state, train_ds, train_labels, batch_size, rng):
    """Train for a single epoch."""
    train_ds_size = len(train_ds)
    steps_per_epoch = train_ds_size // batch_size

    perms = jax.random.permutation(rng, len(train_ds))
    # skip incomplete batch to avoid a recompile of apply_model
    perms = perms[: steps_per_epoch * batch_size]
    perms = perms.reshape((steps_per_epoch, batch_size))

    epoch_loss = []
    epoch_accuracy = []

    for perm in perms:
        batch_images = train_ds[perm, ...]
        batch_labels = train_labels[perm, ...]
        grads, loss, accuracy = apply_model(state, batch_images, batch_labels)
        state = update_model(state, grads)
        epoch_loss.append(loss)
        epoch_accuracy.append(accuracy)

    train_loss = np.mean(epoch_loss)
    train_accuracy = np.mean(epoch_accuracy)
    return state, train_loss, train_accuracy


@step
def preprocess(data: ArrayMapping) -> ArrayMapping:
    """
    Expand dimensions of images.

    Parameters
    ----------
    data: ArrayMapping
        Raw input dataset, as a compressed NumPy array collection.

    Returns
    -------
    ArrayMapping
        Dataset with expanded dimensions.
    """

    data["x_train"] = jnp.float32(data["x_train"]) / 255.0
    data["y_train"] = jnp.float32(data["y_train"])
    data["x_test"] = jnp.float32(data["x_test"]) / 255.0
    data["y_test"] = jnp.float32(data["y_test"])

    # add a fake channel axis to make sure images have shape (28, 28, 1)
    if not data["x_train"].shape[-1] == 1:
        data["x_train"] = jnp.expand_dims(data["x_train"], -1)
        data["x_test"] = jnp.expand_dims(data["x_test"], -1)

    return data


@step
def train(data: ArrayMapping) -> tuple[TrainState, ArrayMapping]:
    """Train a ConvNet model on the preprocessed data."""

    x_train, y_train = data["x_train"], data["y_train"]
    x_test, y_test = data["x_test"], data["y_test"]

    train_perm = np.random.permutation(len(x_train))
    train_perm = train_perm[: int(0.5 * len(x_train))]
    train_data, train_labels = x_train[train_perm, ...], y_train[train_perm, ...]

    test_perm = np.random.permutation(len(x_test))
    test_perm = test_perm[: int(0.5 * len(x_test))]
    test_data, test_labels = x_test[test_perm, ...], y_test[test_perm, ...]

    rng = jr.PRNGKey(random.randint(0, 1000))

    # Instantiate the model.
    model = CNN(rngs=nnx.Rngs(0))
    optimizer = nnx.Optimizer(model, optax.adamw(LEARNING_RATE, MOMENTUM))
    metrics = nnx.MultiMetric(
    accuracy=nnx.metrics.Accuracy(),
    loss=nnx.metrics.Average('loss'),
    )

    for epoch in range(EPOCHS):
        rng, input_rng = jax.random.split(rng)
        state, train_loss, train_accuracy = train_epoch(
            state, train_data, train_labels, BATCH_SIZE, input_rng
        )

    data = {
        "x_train": train_data,
        "y_train": train_labels,
        "x_test": test_data,
        "y_test": test_labels,
    }

    return state, data


@step
def benchmark_model(state: TrainState, data: ArrayMapping) -> None:
    """Evaluate the model and log metrics in nnbench."""

    # the nnbench portion.
    benchmarks = nnbench.collect(HERE)
    params = MNISTTestParameters(params=state.params, data=data)
    result = nnbench.run(benchmarks, name="nnbench-mnist-run", params=params)

    # Log metrics to the step.
    log_metadata(metadata=result.to_json())


@pipeline
def mnist_jax():
    """Load MNIST data, train, and benchmark a simple ConvNet model."""
    mnist = load_mnist()
    mnist = preprocess(mnist)
    state, data = train(mnist)
    benchmark_model(state, data)


if __name__ == "__main__":
    mnist_jax()
