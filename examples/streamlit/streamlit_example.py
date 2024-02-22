import asyncio
import base64
import pickle
from pathlib import Path
from typing import Any

import pandas as pd
import prefect
import streamlit as st
from prefect import deployments
from prefect.results import PersistedResultBlob

from nnbench import types

LOCAL_PREFECT_URL = "http://127.0.0.1:4200"
LOCAL_PREFECT_PERSISTENCE_FOLDER = Path.home() / ".prefect" / "storage"
prefect_client = prefect.PrefectClient(api=LOCAL_PREFECT_URL)

if "benchmarks" not in st.session_state:
    st.session_state["benchmarks"] = []


def setup_ui():
    st.title("Streamlit & Prefect Demo")
    bm_params = {
        "random_state": st.number_input("Random State", value=42, step=1),
        "n_features": st.number_input("N Features", value=2, step=1),
        "n_samples": st.number_input("N Samples", value=100, step=5),
        "noise": st.number_input("Noise", value=0.2, format="%f"),
    }
    return bm_params


async def run_bms(params: dict[str, Any]) -> str:
    result = await deployments.run_deployment(
        "train-and-benchmark/benchmark-runner", parameters={"data_params": params}
    )
    return result.state.result().storage_key


def get_bm_artifacts(storage_key: str) -> None:
    blob_path = LOCAL_PREFECT_PERSISTENCE_FOLDER / storage_key
    blob = PersistedResultBlob.parse_raw(blob_path.read_bytes())
    bm_records: tuple[types.BenchmarkRecord, ...] = pickle.loads(base64.b64decode(blob.data))

    bms = [pd.DataFrame(record.benchmarks) for record in bm_records]
    for df in bms:
        df["value"] = df["value"].apply(lambda x: f"{x:.2e}")
    cxs = [pd.DataFrame([record.context.data]) for record in bm_records]

    display_data = [bms + [cxs[0]]]  # Only need context once
    st.session_state["benchmarks"].extend(display_data)


if __name__ == "__main__":
    bm_params = setup_ui()
    if st.button("Run Benchmarks"):
        storage_key = asyncio.run(run_bms(params=bm_params))
        get_bm_artifacts(storage_key)
        st.write("Benchmark Results")
        for i, benchmark in reversed(list(enumerate(st.session_state["benchmarks"]))):
            with st.expander(f"Benchmark Run {i+1}"):
                meta, metric, ctx = benchmark
                st.write("Model Attributes")
                st.table(meta)
                st.write("Model Metrics")
                st.table(metric)
                st.write("Context Configuration")
                st.table(ctx)
