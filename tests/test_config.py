import logging
import os
from pathlib import Path

import pytest

from nnbench.config import NNBenchConfig, parse_nnbench_config

empty = NNBenchConfig.from_toml({})

test_toml = """
[tool.nnbench]
log-level = "DEBUG"

[tool.nnbench.context.myctx]
name = "myctx"
classpath = "nnbench.context.PythonInfo"
arguments = { packages = ["rich", "pyyaml"] }
"""

test_toml_with_unknown_key = (
    test_toml
    + """

[tool.nnbench.what]
hello = "world"
"""
)


def test_config_load_and_parse(tmp_path: Path) -> None:
    tmp_pyproject = tmp_path / "pyproject.toml"
    tmp_pyproject.write_text(test_toml)

    cfg = parse_nnbench_config(tmp_pyproject)
    assert cfg.log_level == "DEBUG"
    assert len(cfg.context) == 1
    assert cfg.context[0].name == "myctx"


def test_config_load_with_unknown_key(tmp_path: Path, caplog: pytest.LogCaptureFixture) -> None:
    tmp_pyproject = tmp_path / "pyproject.toml"
    tmp_pyproject.write_text(test_toml_with_unknown_key)

    # if this doesn't crash, we know that the unknown key does not make it into the config.
    cfg = parse_nnbench_config(tmp_pyproject)
    assert cfg != empty

    # autodiscovery with no config available should fail.
    with caplog.at_level(logging.DEBUG):
        tmp_pyproject.unlink()
        os.chdir(tmp_path)
        cfg = parse_nnbench_config()
        assert cfg == empty
        assert "could not locate pyproject.toml" in caplog.text
