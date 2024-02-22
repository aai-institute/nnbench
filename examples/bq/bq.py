from google.cloud import bigquery

import nnbench
from nnbench.context import GitEnvironmentInfo


def main():
    client = bigquery.Client()

    # TODO: Fill these out with your appropriate resource names.
    table_id = "<PROJECT>.<DATASET>.<TABLE>"

    job_config = bigquery.LoadJobConfig(
        autodetect=True, source_format=bigquery.SourceFormat.NEWLINE_DELIMITED_JSON
    )

    runner = nnbench.BenchmarkRunner()
    res = runner.run("benchmarks.py", params={"a": 1, "b": 1}, context=(GitEnvironmentInfo(),))

    load_job = client.load_table_from_json(
        res.compact(mode="flatten", sep="_"), table_id, job_config=job_config
    )
    load_job.result()


if __name__ == "__main__":
    main()
