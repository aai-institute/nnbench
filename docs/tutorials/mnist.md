# Integrating nnbench into an existing ML pipeline

Thanks to nnbench's modularity, we can easily integrate it into existing ML experiment code.

As an example, we use an MNIST pipeline written for the popular ML framework [JAX](https://jax.readthedocs.io/en/latest/).
While the actual data sourcing and training code is interesting on its own, we focus solely on the nnbench application part.
You can find the full example code in the nnbench [repository.](https://github.com/aai-institute/nnbench/tree/main/examples/mnist)

## Defining and organizing benchmarks

To properly structure our project, we avoid mixing training pipeline code and benchmark code by placing all benchmarks in a standalone file, similarly to how you might structure unit tests for your code.

```python
--8<-- "examples/mnist/benchmarks.py"
```

This definition is short and sweet, and contains a few important details:

* Both functions are given the `@nnbench.benchmark` decorator - this enables our runner to find and collect them before starting the benchmark run.
* The `modelsize` benchmark is given a custom name (`"Model size (MB)"`), indicating that the resulting number is the combined size of the model weights in megabytes.
This is done for display purposes, to improve interpretability when reporting results.
* The `params` argument is the same in both benchmarks, both in name and type. This is important, since it ensures that both benchmarks will be run with the same model weights.

That's all - now we can shift over to our main pipeline code and see what is necessary to execute the benchmarks and visualize the results.

## Setting up a benchmark runner and parameters

After finishing the benchmark setup, we only need a few more lines to augment our pipeline.

We assume that the benchmark file is located in the same folder as the training pipeline - thus, we can specify our parent directory as the place in which to search for benchmarks:

```python
--8<-- "examples/mnist/mnist.py:26:26"
```

Next, we can define a custom subclass of `nnbench.Params` to hold our benchmark parameters.
Benchmark parameters are a set of variables used as inputs to the benchmark functions collected during the benchmark run.

Since our benchmarks above are parametrized by the model weights (named `params` in the function signatures) and the MNIST data split (called `data`), we define our parameters to take exactly these two values.

```python
--8<-- "examples/mnist/mnist.py:38:41"
```

And that's it! After we implement all training code, we just run nnbench directly after training in our top-level pipeline function:

```python
--8<-- "examples/mnist/mnist.py:213:223"
```

We use `to="console"` as a keyword argument to `BenchmarkRunner.report()` to print the results directly to the terminal.
Notice how by we can reuse the training artifacts in nnbench as parameters to obtain results right after training!

The output might look like this:

```
name               value
---------------  -------
accuracy         0.9712
Model size (MB)  3.29783
```

This can be improved in a number of ways - for example by enriching it with metadata about the model architecture, the used GPU, etc.
For more information on how to supply context to benchmarks, check the [user guide](../guides/index.md) section.
