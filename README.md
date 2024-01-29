# nnbench: A small framework for benchmarking machine learning models

Welcome to nnbench, a framework for benchmarking machine learning models.
The main goals of this project are:

1. To provide a portable, easy-to-use solution for model evaluation that leads to better ML experiment organization, and
2. To integrate with experiment and metadata tracking solutions for easy adoption.

On a high level, you can think of nnbench as "pytest for ML models" - you define benchmarks similarly to test cases, collect them, and selectively run them based on model type, markers, and environment info.

What's new is that upon completion, you can stream the resulting data to any sink of your choice (including multiple at the same), which allows easy integration with experiment trackers and metadata stores.

See the `quickstart.md` for usage examples.

## Installation

⚠️ nnbench is an experimental project - expect bugs and sharp edges.

Install it directly from source, for example either using `pip` or `poetry`:

```shell
pip install git+https://github.com/aai-institute/nnbench.git
# or
poetry add git+https://github.com/aai-institute/nnbench.git
```

A PyPI package release is planned for the future.


## Contributing

We encourage and welcome contributions from the community to enhance the project.
Please check [discussions](https://github.com/aai-institute/nnbench/discussions) or raise an [issue](https://github.com/aai-institute/nnbench/issues) on GitHub for any problems you encounter with the library.

For information on the general development workflow, see the [contribution guide](CONTRIBUTING.md).

## License

The nnbench library is distributed under the [Apache-2 license](LICENSE).
