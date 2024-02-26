# Querying benchmark results at scale with duckDB

For a powerful way to query, filter, and visualize benchmark records, [duckdb](https://duckdb.org/) is a great choice.
This page contains a quick tutorial for analyzing benchmark results with duckDB.

## Prerequisites and installation

To use duckdb, install the duckdb Python package by running `pip install --upgrade duckdb`.
In this tutorial, we are going to be using the in-memory database only, but you can easily persist SQL views of records on disk as well.

## Writing and ingesting benchmark records

We consider the following easy benchmark example:

```python
--8<-- "examples/bq/benchmarks.py"
```

Running both of these benchmarks produces a benchmark record, which we can save to disk using the `FileIO` class.

```python
import nnbench
from nnbench.context import GitEnvironmentInfo
from nnbench.reporter.file import FileIO

runner = nnbench.BenchmarkRunner()
record = runner.run("benchmarks.py", params={"a": 1, "b": 1}, context=(GitEnvironmentInfo(),))

fio = FileIO()
fio.write(record, "record.json", driver="ndjson")
```

This writes a newline-delimited JSON file as `record.json` into the current directory. We choose this format because it is ideal for duckdb to work with.

Now, we can easily ingest the record into a duckDB database:

```python
import duckdb

duckdb.sql(
    """
    SELECT name, value FROM read_ndjson_auto('record.json')
    """
).show()

# ----- prints: -----
# ┌─────────┬───────┐
# │  name   │ value │
# │ varchar │ int64 │
# ├─────────┼───────┤
# │ prod    │     1 │
# │ sum     │     2 │
# └─────────┴───────┘
```

## Querying metadata directly in SQL by flattening the context struct

By default, the benchmark context struct, which holds additional information about the benchmark runs, is inlined into the raw dictionary before saving it to a file.
This is not ideal for some SQL implementations, where you might not be able to filter records easily by interpreting the serialized `context` struct.

To improve, you can pass `ctxmode="flatten"` to the `FileIO.write()` method to flatten the context and inline all nested values instead.
This comes at the expense of an inflated schema, i.e. more columns in the database.

```python
fio = FileIO()
fio.write(record, "record.json", driver="ndjson", ctxmode="flatten")
```

In the example above, we used the `GitEnvironmentInfo` context provider to log some information on the git environment we ran our benchmarks in.
In flat mode, this includes the `git.commit` and `git.repository` values, telling us at which commit and in which repository the benchmarks were run, respectively.

To log this information in a duckDB view, we run the following on a flat-context NDJSON record:

```python
duckdb.sql(
    """
    SELECT name, value, \"git.commit\", \"git.repository\" FROM read_ndjson_auto('record.json')
    """
).show()

#   ---------------------------------- prints ------------------------------------------
# ┌─────────┬───────┬──────────────────────────────────────────┬───────────────────────┐
# │  name   │ value │                git.commit                │    git.repository     │
# │ varchar │ int64 │                 varchar                  │        varchar        │
# ├─────────┼───────┼──────────────────────────────────────────┼───────────────────────┤
# │ prod    │     1 │ 0d47d7bcd2d2c13b69796355fe9d4ef5f50b1edb │ aai-institute/nnbench │
# │ sum     │     2 │ 0d47d7bcd2d2c13b69796355fe9d4ef5f50b1edb │ aai-institute/nnbench │
# └─────────┴───────┴──────────────────────────────────────────┴───────────────────────┘
```

This method is great to select a subset of context values when the full context metadata structure is not required.

!!! Tip
    In duckDB specifically, this is equivalent to dotted access of the "context" column if `ctxmode="inline"`.
    This means that the following also works to obtain the git commit mentioned above:
    ```python
    duckdb.sql(
        """
        SELECT name, value, context.git.commit AS \"git.commit\" FROM read_ndjson_auto('record.json')
        """
    ).show()
    ```
