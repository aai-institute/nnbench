import time
from pathlib import Path

from nnbench.types import PathArtifactLoader


def test_load_local_file(local_file: Path) -> None:
    loader = PathArtifactLoader()
    loaded_path: Path = loader.load(local_file)
    assert loaded_path.exists()
    assert loaded_path.read_text() == "Test content"


def test_caching_PathArtifactLoader(local_file: Path) -> None:
    artifact_loader = PathArtifactLoader()

    # Measure the time for the first load
    start_time_first_load = time.perf_counter()
    artifact_loader.load(local_file)
    end_time_first_load = time.perf_counter()

    # Measure the time for the second, cached load
    start_time_second_load = time.perf_counter()
    artifact_loader.load(local_file)
    end_time_second_load = time.perf_counter()

    first_load_duration = end_time_first_load - start_time_first_load
    second_load_duration = end_time_second_load - start_time_second_load
    assert second_load_duration < first_load_duration
