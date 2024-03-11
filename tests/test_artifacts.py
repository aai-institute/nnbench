from pathlib import Path

from nnbench.types import FilePathArtifactLoader


def test_load_local_file(local_file: Path, tmp_path: Path) -> None:
    test_dir = tmp_path / "test_load_dir"
    loader = FilePathArtifactLoader(local_file, test_dir)
    loaded_path: Path = loader.load()
    assert loaded_path.exists()
    assert loaded_path.read_text() == "Test content"
