# Streaming benchmarks to a cloud database

Once you obtain the results of your benchmarks, you will most likely want to store them somewhere.
Whether that is in storage as flat files, on a server, or in a database, `nnbench` allows you to write records anywhere, provided the destination supports JSON.

This is a small guide containing a snippet on how to stream benchmark results to a Google Cloud BigQuery table.

## The benchmarks

Configure your benchmarks as normal, for example by separating them into a Python file.
The following is a very simple example benchmark setup.

```python
--8<-- "examples/bq/benchmarks.py"
```

## Setting up a BigQuery client

In order to authenticate with BigQuery, follow the official [Google Cloud documentation](https://cloud.google.com/bigquery/docs/authentication#client-libs).
In this case, we rely on Application Default Credentials (ADC), which can be configured with the `gcloud` CLI.

To interact with BigQuery from Python, the `google-cloud-bigquery` package has to be installed.
You can do this e.g. using pip via `pip install --upgrade google-cloud-bigquery`.

## Creating a table

Within your configured project, proceed by creating a destination table to write the benchmarks to.
Consider the [BigQuery Python documentation on tables](https://cloud.google.com/bigquery/docs/tables#python) for how to create a table programmatically.

!!! Note
    If the configured dataset does not exist, you will have to create it as well, either programmatically via the `bigquery.Client.create_dataset` API or in the Google Cloud console.

## Using BigQuery's schema auto-detection

In order to skip tedious schema inference by hand, we can use BigQuery's [schema auto-detection from JSON records](https://cloud.google.com/bigquery/docs/schema-detect).
All we have to do is configure a BigQuery load job to auto-detect the schema from the Python dictionaries in memory:

```python
--8<-- "examples/bq/bq.py:13:16"
```

After that, write and stream the compacted benchmark record directly to your destination table.
In this example, we decide to flatten the benchmark context to be able to extract scalar context values directly from the result table using raw SQL queries.
Note that you have to use a custom separator (an underscore `"_"` in this case) for the context data, since BigQuery does not allow dots in column names.

```python
--8<-- "examples/bq/bq.py:21:25"
```

!!! Tip
    If you would like to save the context dictionary as a struct instead, use `mode = "inline"` in the call to `BenchmarkRecord.compact()`.

And that's all! To check that the records appear as expected, you can now query the data e.g. like so:

```python
# check that the insert worked.
query = f'SELECT name, value, time_ns, git_commit AS commit FROM {table_id}'
r = client.query(query)
for row in r.result():
    print(r)
```

## Recap and the full source code

In this tutorial, we

1) defined and ran a benchmark workload using `nnbench`.
2) configured a Google Cloud BigQuery client and a load job to insert benchmark records into a table, and
3) inserted the records into the destination table.

The full source code for this tutorial is included below, and also in the [nnbench repository](https://github.com/aai-institute/nnbench/tree/main/examples/bq).

```python
--8<-- "examples/bq/bq.py"
```
