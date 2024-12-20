# Contributing to nnbench

Thank you for your interest in contributing to this project!

We appreciate issue reports, pull requests for code and documentation,
as well as any project-related communication through [GitHub Discussions](https://github.com/aai-institute/nnbench/discussions).

## Getting Started

To get started with development, you can follow these steps:

1. Clone this repository:

    ```shell
    git clone https://github.com/aai-institute/nnbench.git
    ```

2. Navigate to the directory and install the development dependencies into a virtual environment, e.g. using `uv`:

    ```shell
    cd nnbench
    uv venv --seed -p 3.11
    source .venv/bin/activate
    ```

3. After making your changes, verify they adhere to our Python code style by running `pre-commit`:
    
    ```shell
    uvx pre-commit run --all-files --verbose --show-diff-on-failure
    ```

    You can also set up Git hooks through `pre-commit` to perform these checks automatically:
    
    ```shell
    pre-commit install
    ```

4. To run the tests, just invoke `pytest` from the package root directory:
    ```shell
    uv run pytest -s
    ```

## Updating dependencies

Dependencies should stay locked for as long as possible, ideally for a whole release.
If you have to update a dependency during development, you should do the following:

1. If it is a core dependency needed for the package, add it to the `dependencies` section in the `pyproject.toml` via `uv add <dep>`.
2. In case of a development dependency, add it to the `dev` section of the `project.dependency-groups` table instead (`uv add --group dev <dep>`).
3. Dependencies needed for documentation generation are found in the `docs` sections of `project.dependency-groups` (`uv add --group docs <dep>`).

After adding the dependency in either of these sections, use `uv lock` to pin all dependencies again:

```shell
uv lock
```

> [!IMPORTANT]
> Since the official development version is Python 3.11, please run the above commands in a virtual environment with Python 3.11.
