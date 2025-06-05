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
usage: nnbench run [-h] [-n <name>] [-j <N>] [--context <key=value>] [-t <tag>] [-o <file>] [--jsonifier <classpath>] [<benchmarks>]

positional arguments:
  <benchmarks>          A Python file or directory of files containing benchmarks to run.

options:
  -h, --help            show this help message and exit
  -n, --name <name>     A name to assign to the benchmark run, for example for record keeping in a database.
  -j <N>                Number of processes to use for running benchmarks in parallel, default: -1 (no parallelism)
  --context <key=value>
                        Additional context values giving information about the benchmark run.
  -t, --tag <tag>       Only run benchmarks marked with one or more given tag(s).
  -o, --output-file <file>
                        File or stream to write results to, defaults to stdout.
  --jsonifier <classpath>
                        Function to create a JSON representation of input parameters with, helping make runs reproducible.
```

To run a benchmark workload contained in a single `benchmarks.py` file, you would run `nnbench run benchmarks.py`.
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

### Streaming results to different locations with URIs

Like in the nnbench Python SDK, you can use builtin and custom reporters to write benchmark results to various locations.
To select a reporter implementation, specify the output file as a URI, with a leading protocol, and suffixed with `://`.

!!! Example
    The builtin file reporter supports multiple file system protocols of the `fsspec` project.
    If you have `fsspec` installed, you can stream benchmark results to different cloud storage providers like so:

    ```commandline
    # Write a result to disk...
    nnbench run benchmarks.py -o result.json
    # ...or to an S3 storage bucket...
    nnbench run benchmarks.py -o s3://my-bucket/result.json
    # ...or to GCS...
    nnbench run benchmarks.py -o gs://my-bucket/result.json
    # ...or lakeFS:
    nnbench run benchmarks.py -o lakefs://my-repo/my-branch/result.json
    ```

    For a comprehensive list of supported protocols, see the [fsspec documentation](https://filesystem-spec.readthedocs.io/en/latest/api.html#other-known-implementations).

    If `fsspec` is not installed, only local files can be written (i.e. to the executing machine's filesystem).

## Comparing results across multiple benchmark runs

To create a comparison table between multiple benchmark runs, use the `nnbench compare` command.

```commandline
$ nnbench compare -h
usage: nnbench compare [-h] [--comparison-file <JSON>] [-C <name>] [-E <name>] results [results ...]

positional arguments:
  results               Results to compare. Can be given as local files or remote URIs.

options:
  -h, --help            show this help message and exit
  --comparison-file <JSON>
                        A file containing comparison functions to run on benchmarking metrics.
  -C, --include-context <name>
                        Context values to display in the comparison table. Use dotted syntax for nested context values.
  -E, --extra-column <name>
                        Additional result data to display in the comparison table.
```

Supposing we have the following results from previous runs, for a benchmark `add(a,b)` that adds two integers:

```json
// Pretty-printed JSON, obtained as jq . result1.json
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
      "timestamp": 1733157676,
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
// jq . result2.json
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
      "timestamp": 1733157724,
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

we can compare them in a table view by running `nnnbench compare result1.json result2.json`:

```commandline
$ nnbench compare result1.json result2.json
┏━━━━━━━━━━━━━━━━━━┳━━━━━┓
┃ Benchmark run    ┃ add ┃
┡━━━━━━━━━━━━━━━━━━╇━━━━━┩
│ nnbench-3ff188b4 │ 300 │
│ nnbench-5cbb85f8 │ 300 │
└──────────────────┴─────┘
```

To include context values in the table - in our case, we might want to display the `foo` value - use the `-C` switch (you can use it multiple times to include multiple values):

```commandline
$ nnbench compare result1.json result2.json -C foo
┏━━━━━━━━━━━━━━━━━━┳━━━━━┳━━━━━┓
┃ Benchmark run    ┃ add ┃ foo ┃
┡━━━━━━━━━━━━━━━━━━╇━━━━━╇━━━━━┩
│ nnbench-3ff188b4 │ 300 │ bar │
│ nnbench-5cbb85f8 │ 300 │ baz │
└──────────────────┴─────┴─────┘
```

To learn how to define per-metric comparisons and use comparisons in a continuous training pipeline, refer to the [comparison documentation](comparisons.md).
