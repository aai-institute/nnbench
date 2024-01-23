import logging
from pathlib import Path

import pytest

HERE = Path(__file__).parent

logger = logging.getLogger("nnbench")
logger.setLevel(logging.DEBUG)


@pytest.fixture(scope="session")
def testfolder() -> str:
    """A test directory for benchmark collection."""
    return str(HERE / "testproject")


# TODO: Consider merging all test directories into one,
#  filtering benchmarks by testcase via tags.
@pytest.fixture(scope="session")
def typecheckfolder() -> str:
    return str(HERE / "typechecking")
