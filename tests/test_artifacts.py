from pathlib import Path

from nnbench.types import PathArtifactLoader


def test_load_local_file(local_file: Path) -> None:
    loader = PathArtifactLoader()
    loaded_path: Path = loader.load(local_file)
    assert loaded_path.exists()
    assert loaded_path.read_text() == "Test content"
