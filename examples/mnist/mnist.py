"""
JAX MNIST example with nnbench.

Demonstrates the use of nnbench to collect and log metrics right after training.

Source: https://github.com/google/flax/blob/main/examples/mnist
"""

import random
from dataclasses import dataclass
from pathlib import Path
from typing import Mapping

import flax.linen as nn
import fsspec
import jax
import jax.numpy as jnp
import jax.random as jr
import numpy as np
import optax
from flax.training.train_state import TrainState

import nnbench
from nnbench.runner import BenchmarkRunner

HERE = Path(__file__).parent

ArrayMapping = dict[str, jax.Array | np.ndarray]

INPUT_SHAPE = (28, 28, 1)  # H x W x C (= 1, BW grayscale images)
NUM_CLASSES = 10
BATCH_SIZE = 128
EPOCHS = 1
LEARNING_RATE = 0.1
MOMENTUM = 0.9


@dataclass(frozen=True)
class MNISTTestParameters(nnbench.Params):
    params: Mapping[str, jax.Array]
    data: ArrayMapping


class ConvNet(nn.Module):
    @nn.compact
    def __call__(self, x):
        x = nn.Conv(features=32, kernel_size=(3, 3))(x)
        x = nn.relu(x)
        x = nn.avg_pool(x, window_shape=(2, 2), strides=(2, 2))
        x = nn.Conv(features=64, kernel_size=(3, 3))(x)
        x = nn.relu(x)
        x = nn.avg_pool(x, window_shape=(2, 2), strides=(2, 2))
        x = x.reshape(x.shape[0], -1)  # flatten
        x = nn.Dense(features=256)(x)
        x = nn.relu(x)
        x = nn.Dense(features=NUM_CLASSES)(x)
        return x


@jax.jit
def apply_model(state, images, labels):
    """Computes gradients, loss and accuracy for a single batch."""

    def loss_fn(params):
        logits = state.apply_fn({"params": params}, images)
        one_hot = jax.nn.one_hot(labels, 10)
        loss = jnp.mean(optax.softmax_cross_entropy(logits=logits, labels=one_hot))
        return loss, logits

    grad_fn = jax.value_and_grad(loss_fn, has_aux=True)
    (loss, logits), grads = grad_fn(state.params)
    accuracy = jnp.mean(jnp.argmax(logits, -1) == labels)
    return grads, loss, accuracy


@jax.jit
def update_model(state, grads):
    return state.apply_gradients(grads=grads)


def create_train_state(rng):
    """Creates initial `TrainState`."""
    convnet = ConvNet()
    params = convnet.init(rng, jnp.ones([1, *INPUT_SHAPE]))["params"]
    tx = optax.sgd(learning_rate=LEARNING_RATE, momentum=MOMENTUM)
    return TrainState.create(apply_fn=convnet.apply, params=params, tx=tx)


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

    baseurl = "http://yann.lecun.com/exdb/mnist/"

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
    rng, init_rng = jr.split(rng)
    state = create_train_state(init_rng)

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


def mnist_jax():
    """Load MNIST data and train a simple ConvNet model."""
    mnist = load_mnist()
    mnist = preprocess(mnist)
    state, data = train(mnist)

    # the nnbench portion.
    runner = BenchmarkRunner()
    params = MNISTTestParameters(params=state.params, data=data)
    result = runner.run(HERE, params=params)
    runner.report(to="console", result=result)


if __name__ == "__main__":
    mnist_jax()
