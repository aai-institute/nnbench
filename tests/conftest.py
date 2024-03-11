import logging
from pathlib import Path

import pytest

HERE = Path(__file__).parent

logger = logging.getLogger("nnbench")
logger.setLevel(logging.DEBUG)


@pytest.fixture(scope="session")
def testfolder() -> str:
    """A test directory for benchmark collection."""
    return str(HERE / "benchmarks")


@pytest.fixture
def local_file(tmp_path: Path) -> Path:
    file_path = tmp_path / "test_file.txt"
    file_path.write_text("Test content")
    return file_path
