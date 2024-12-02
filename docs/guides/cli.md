# The `nnbench` command-line interface (CLI)

While you can always use nnbench to directly run your benchmarks in your Python code, for example as part of a workflow, there is also the option of running benchmarks from the command line.
This way of using nnbench is especially useful for integrating into a machine learning pipeline as part of a continuous training/delivery scenario.

## General options

The `nnbench` CLI has the following top-level options:

```commandline
$ nnbench
usage: nnbench [-h] [--version] [--log-level <level>]  ...

options:
  -h, --help           show this help message and exit
  --version            show program's version number and exit
  --log-level <level>  Log level to use for the nnbench package, defaults to NOTSET (no logging).

Available commands:
  
    run                Run a benchmark workload.
    compare            Compare results from multiple benchmark runs.
```

Supported log levels are `"DEBUG", "INFO", "WARNING", "ERROR"`, and `"CRITICAL"`.

## Running benchmark workloads on the command line

This is the responsibility of the `nnbench run` subcommand.

```commandline
$ nnbench run -h                                                                         
usage: nnbench run [-h] [--context <key=value>] [-t <tag>] [-o <file>] [<benchmarks>]

positional arguments:
  <benchmarks>          Python file or directory of files containing benchmarks to run.

options:
  -h, --help            show this help message and exit
  --context <key=value>
                        Additional context values giving information about the benchmark run.
  -t, --tag <tag>       Only run benchmarks marked with one or more given tag(s).
  -o, --output-file <file>
                        File or stream to write results to, defaults to stdout.
```

So to run a benchmark workload contained in a single `benchmarks.py` file, you would run `nnbench run benchmarks.py`.
For tips on how to structure and annotate your benchmarks, refer to the [organization](organization.md) guide.

For injecting context values on the command line, you need to give the key-value pair explicitly by passing the `--context` switch.
For example, to look up and persist the `pyyaml` version in the current environment, you could run the following:

```commandline
nnbench run .sandbox/example.py --context=pyyaml=`python3 -c "from importlib.metadata import version; print(version('pyyaml'))"`
```

!!! tip
    Both `--context` and `--tag` are appending options, so you can pass multiple context values and multiple tags per run.

## Comparing results across multiple benchmark runs

To create a comparison table between multiple benchmark runs, use the `nnbench compare` command.

```commandline
$ nnbench compare -h                                            
usage: nnbench compare [-h] [-P <name>] [-C <name>] [-E <name>] records [records ...]

positional arguments:
  records               Records to compare results for. Can be given as local files or remote URIs.

options:
  -h, --help            show this help message and exit
  -P, --include-parameter <name>
                        Names of input parameters to display in the comparison table.
  -C, --include-context <name>
                        Context values to display in the comparison table. Use dotted syntax for nested context values.
  -E, --extra-column <name>
                        Additional record data to display in the comparison table.
```

Supposing we have the following records from previous runs:

```json
// Contents of record1.json:
{
  "run": "nnbench-f0822b3c",
  "context": {
    "foo": "bar"
  },
  "benchmarks": [
    {
      "name": "add",
      "function": "add",
      "description": "",
      "date": "2024-11-26T14:04:15",
      "error_occurred": false,
      "error_message": "",
      "parameters": {
        "a": 50,
        "b": 25
      },
      "value": 75,
      "time_ns": 917
    },
    {
      "name": "mul",
      "function": "mul",
      "description": "",
      "date": "2024-11-26T14:04:15",
      "error_occurred": false,
      "error_message": "",
      "parameters": {
        "a": 50,
        "b": 25
      },
      "value": 1250,
      "time_ns": 625
    },
    {
      "name": "sub",
      "function": "sub",
      "description": "",
      "date": "2024-11-26T14:04:15",
      "error_occurred": false,
      "error_message": "",
      "parameters": {
        "a": 50,
        "b": 25
      },
      "value": 25,
      "time_ns": 542
    }
  ]
}
```

and

```json
// Contents of record2.json:
{
  "run": "nnbench-4f804c24",
  "context": {
    "foo": "baz"
  },
  "benchmarks": [
    {
      "name": "add",
      "function": "add",
      "description": "",
      "date": "2024-11-26T13:16:48",
      "error_occurred": false,
      "error_message": "",
      "parameters": {
        "a": 50,
        "b": 25
      },
      "value": 75,
      "time_ns": 2541
    },
    {
      "name": "mul",
      "function": "mul",
      "description": "",
      "date": "2024-11-26T13:16:48",
      "error_occurred": false,
      "error_message": "",
      "parameters": {
        "a": 50,
        "b": 25
      },
      "value": 1250,
      "time_ns": 750
    },
    {
      "name": "sub",
      "function": "sub",
      "description": "",
      "date": "2024-11-26T13:16:48",
      "error_occurred": false,
      "error_message": "",
      "parameters": {
        "a": 50,
        "b": 25
      },
      "value": 25,
      "time_ns": 584
    }
  ]
},
```

we can compare them in a table view by running `nnnbench compare record1.json record2.json`:

```commandline
$ nnbench compare record1.json record2.json
┏━━━━━━━━━━━━━━━━━━┳━━━━━┳━━━━━━┳━━━━━┓
┃ Benchmark run    ┃ add ┃ mul  ┃ sub ┃
┡━━━━━━━━━━━━━━━━━━╇━━━━━╇━━━━━━╇━━━━━┩
│ nnbench-f0822b3c │ 75  │ 1250 │ 25  │
│ nnbench-4f804c24 │ 75  │ 1250 │ 25  │
└──────────────────┴─────┴──────┴─────┘
```

To include benchmark parameters in the table, use the `-P` switch (you can supply this multiple times to include multiple parameters):

```commandline
$ nnbench compare record1.json record2.json -P a -P b
┏━━━━━━━━━━━━━━━━━━┳━━━━━┳━━━━━━┳━━━━━┳━━━━━━━━━━━┳━━━━━━━━━━━┓
┃ Benchmark run    ┃ add ┃ mul  ┃ sub ┃ Params->a ┃ Params->b ┃
┡━━━━━━━━━━━━━━━━━━╇━━━━━╇━━━━━━╇━━━━━╇━━━━━━━━━━━╇━━━━━━━━━━━┩
│ nnbench-f0822b3c │ 75  │ 1250 │ 25  │ 50        │ 25        │
│ nnbench-4f804c24 │ 75  │ 1250 │ 25  │ 50        │ 25        │
└──────────────────┴─────┴──────┴─────┴───────────┴───────────┘
```

To include context values in the table - in our case, we might want to display the `foo` value - use the `-C` switch (this is also appending, same as `-P`):

```commandline
$ nnbench compare record1.json record2.json -P a -P b -C foo
┏━━━━━━━━━━━━━━━━━━┳━━━━━┳━━━━━━┳━━━━━┳━━━━━━━━━━━┳━━━━━━━━━━━┳━━━━━┓
┃ Benchmark run    ┃ add ┃ mul  ┃ sub ┃ Params->a ┃ Params->b ┃ foo ┃
┡━━━━━━━━━━━━━━━━━━╇━━━━━╇━━━━━━╇━━━━━╇━━━━━━━━━━━╇━━━━━━━━━━━╇━━━━━┩
│ nnbench-f0822b3c │ 75  │ 1250 │ 25  │ 50        │ 25        │ bar │
│ nnbench-4f804c24 │ 75  │ 1250 │ 25  │ 50        │ 25        │ baz │
└──────────────────┴─────┴──────┴─────┴───────────┴───────────┴─────┘
```

## Configuring the CLI experience in `pyproject.toml`

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

In this case, we are augmenting `nnbench.context.PythonInfo`, a builtin provider, to also collect the versions of the `rich` and `pyyaml` packages from the current environment, and registering it under the name "myctx".

The `name` field is used to register the context provider.
The `classpath` field needs to be a fully qualified Python module path to the context provider class or function.
Any arguments needed to instantiate a context provider class can be given under the `arguments` key in an inline table, which will be passed to the class found under `classpath` as keyword arguments.

Now we can use said provider in a benchmark run using the special `"provider"` key:

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
    This feature is work in progress. The ability to register custom runner, reporter, and comparison classes will be implemented in future releases of nnbench.
