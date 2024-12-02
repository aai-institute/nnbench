# `nnbench` command reference

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
For tips on how to structure and annotate your benchmarks, refer to the [organization](../guides/organization.md) guide.

For injecting context values on the command line, you need to give the key-value pair explicitly by passing the `--context` switch.
For example, to look up and persist the `pyyaml` version in the current environment, you could run the following:

```commandline
nnbench run .sandbox/example.py --context=pyyaml=`python3 -c "from importlib.metadata import version; print(version('pyyaml'))"`
```

!!! tip
    Both `--context` and `--tag` are appending options, so you can pass multiple context values and multiple tags per run.

!!! tip
    For more complex calculations of context values, it is recommended to register a *custom context provider* in your pyproject.toml file.
    An introductory example can be found in the [nnbench CLI configuration guide](pyproject.md).

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

Supposing we have the following records from previous runs, for a benchmark `add(a,b)` that adds two integers:

```json
// Pretty-printed JSON, obtained as <record1.json | jq
{
  "run": "nnbench-3ff188b4",
  "context": {
    "foo": "bar"
  },
  "benchmarks": [
    {
      "name": "add",
      "function": "add",
      "description": "",
      "date": "2024-12-02T17:41:16",
      "error_occurred": false,
      "error_message": "",
      "parameters": {
        "a": 200,
        "b": 100
      },
      "value": 300,
      "time_ns": 1291
    }
  ]
}
```

and

```json
// <record2.json | jq
{
  "run": "nnbench-5cbb85f8",
  "context": {
    "foo": "baz"
  },
  "benchmarks": [
    {
      "name": "add",
      "function": "add",
      "description": "",
      "date": "2024-12-02T17:42:04",
      "error_occurred": false,
      "error_message": "",
      "parameters": {
        "a": 200,
        "b": 100
      },
      "value": 300,
      "time_ns": 1792
    }
  ]
},
```

we can compare them in a table view by running `nnnbench compare record1.json record2.json`:

```commandline
$ nnbench compare record1.json record2.json
┏━━━━━━━━━━━━━━━━━━┳━━━━━┓
┃ Benchmark run    ┃ add ┃
┡━━━━━━━━━━━━━━━━━━╇━━━━━┩
│ nnbench-3ff188b4 │ 300 │
│ nnbench-5cbb85f8 │ 300 │
└──────────────────┴─────┘
```

To include benchmark parameter values in the table, use the `-P` switch (you can supply this multiple times to include multiple parameters).
For example, to see which values were used for `a` and `b` in our `add(a, b)` benchmark above, we supply `-P a` and `-P b`:

```commandline
$ nnbench compare record1.json record2.json -P a -P b
┏━━━━━━━━━━━━━━━━━━┳━━━━━┳━━━━━━━━━━━┳━━━━━━━━━━━┓
┃ Benchmark run    ┃ add ┃ Params->a ┃ Params->b ┃
┡━━━━━━━━━━━━━━━━━━╇━━━━━╇━━━━━━━━━━━╇━━━━━━━━━━━┩
│ nnbench-3ff188b4 │ 300 │ 200       │ 100       │
│ nnbench-5cbb85f8 │ 300 │ 200       │ 100       │
└──────────────────┴─────┴───────────┴───────────┘
```

To include context values in the table - in our case, we might want to display the `foo` value - use the `-C` switch (this is also appending, same as `-P`):

```commandline
$ nnbench compare record1.json record2.json -P a -P b -C foo
┏━━━━━━━━━━━━━━━━━━┳━━━━━┳━━━━━━━━━━━┳━━━━━━━━━━━┳━━━━━┓
┃ Benchmark run    ┃ add ┃ Params->a ┃ Params->b ┃ foo ┃
┡━━━━━━━━━━━━━━━━━━╇━━━━━╇━━━━━━━━━━━╇━━━━━━━━━━━╇━━━━━┩
│ nnbench-3ff188b4 │ 300 │ 200       │ 100       │ bar │
│ nnbench-5cbb85f8 │ 300 │ 200       │ 100       │ baz │
└──────────────────┴─────┴───────────┴───────────┴─────┘
```
