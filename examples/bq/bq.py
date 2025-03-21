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

    benchmarks = nnbench.collect("benchmarks.py")
    res = nnbench.run(benchmarks, params={"a": 1, "b": 1}, context=(GitEnvironmentInfo(),))

    load_job = client.load_table_from_json(res.to_json(), table_id, job_config=job_config)
    load_job.result()


if __name__ == "__main__":
    main()
