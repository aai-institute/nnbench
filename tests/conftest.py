import logging
from pathlib import Path

import pytest

HERE = Path(__file__).parent

logger = logging.getLogger("nnbench")
logger.setLevel(logging.DEBUG)


@pytest.fixture(scope="session")
def testfolder() -> str:
    """A test directory for benchmark collection."""
    return str(HERE / "test_benchmarks")
