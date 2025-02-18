# Configuring the CLI experience in `pyproject.toml`

To create your custom CLI profile for nnbench, you can set certain options directly in your `pyproject.toml` file.
Like other tools, nnbench will look for a `[tool.nnbench]` table inside the pyproject.toml file, and if found, use it to set certain values.

Currently, you can set the log level, and register custom context provider classes.

### General options

```toml
[tool.nnbench]
# This sets the `nnbench` logger's level to "DEBUG", enabling debug log collections.
log-level = "DEBUG"
```

### Registering custom context providers

As a quick refresher, in nnbench, a *context provider* is a function taking no arguments, and returning a Python dictionary with string keys:

```python
import os

def foo() -> dict[str, str]:
    """Returns a context value named 'foo', containing the value of the FOO environment variable."""
    return {"foo": os.getenv("FOO", "")}
```

If you would like to use a custom context provider to collect metadata before a CLI benchmark run, you can give its details in a `[tool.nnbench.context]` table.

```toml
[tool.nnbench.context.myctx]
name = "myctx"
classpath = "nnbench.context.PythonInfo"
arguments = { packages = ["rich", "pyyaml"] }
```

In this case, we are augmenting `nnbench.context.PythonInfo`, a builtin provider class, to also collect the versions of the `rich` and `pyyaml` packages from the current environment, and registering it under the name "myctx".

The `name` field is used to register the context provider.
The `classpath` field needs to be a fully qualified Python module path to the context provider class or function.
Any arguments needed to instantiate a context provider class can be given under the `arguments` key in an inline table, which will be passed to the class found under `classpath` as keyword arguments.

!!! Warning
    If you register a context provider *function*, you **must** leave the `arguments` key out of the above TOML table, since by definition, context providers do not take any arguments in their `__call__()` signature.

Now we can use said provider in a benchmark run by passing the special `"provider"` key:

```commandline
$ nnbench run benchmarks.py --context=provider=myctx
Context values:
{
    "python": {
        "version": "3.11.10",
        "implementation": "CPython",
        "buildno": "main",
        "buildtime": "Sep  7 2024 01:03:31",
        "packages": {
            "rich": "13.9.3",
            "pyyaml": "6.0.2"
        }
    }
}

<tabular result output>
```

!!! Tip
    This feature is a work in progress. The ability to register custom IO and comparison classes will be implemented in future releases of nnbench.
