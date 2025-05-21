import jax
import jax.numpy as jnp
from mnist import ArrayMapping, ConvNet

import nnbench


@nnbench.benchmark
def accuracy(params: ArrayMapping, data: ArrayMapping) -> float:
    x_test, y_test = data["x_test"], data["y_test"]

    cn = ConvNet()
    y_pred = cn.apply({"params": params}, x_test)
    return jnp.mean(jnp.argmax(y_pred, -1) == y_test).item()


@nnbench.benchmark(name="Model size (MB)")
def modelsize(params: ArrayMapping) -> float:
    nbytes = sum(x.size * x.dtype.itemsize for x in jax.tree_util.tree_leaves(params))
    return nbytes / 1e6
