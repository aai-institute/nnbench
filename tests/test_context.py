import os
from pathlib import Path

from nnbench.context import CPUInfo, GitEnvironmentInfo, PythonInfo


def test_cpu_info_provider() -> None:
    """Tests CPU info integrity, along with some assumptions about metrics."""

    c = CPUInfo()
    ctx = c()["cpu"]
    for k in [
        "architecture",
        "system",
        "frequency",
        "min_frequency",
        "max_frequency",
        "frequency_unit",
        "memory_unit",
    ]:
        assert k in ctx

    assert isinstance(ctx["frequency"], float)
    assert isinstance(ctx["min_frequency"], float)
    assert isinstance(ctx["max_frequency"], float)
    assert isinstance(ctx["total_memory"], float)


def test_git_info_provider() -> None:
    """Tests git provider value integrity, along with some data sanity checks."""
    g = GitEnvironmentInfo()
    # git info needs to be collected inside the nnbench repo, otherwise we get no values.
    os.chdir(Path(__file__).parent)
    ctx = g()["git"]

    # tag is not checked, because that can be empty (e.g. in a shallow repo clone).
    for k in ["provider", "repository", "commit"]:
        assert k in ctx
        assert ctx[k] != "", f"empty value for context {k!r}"

    assert ctx["repository"].split("/")[1] == "nnbench"
    assert ctx["provider"] == "github.com"


def test_python_info_provider() -> None:
    """Tests Python info, along with an example of Python package version scraping."""
    packages = ["rich", "pytest"]
    p = PythonInfo(packages=packages)
    ctx = p()["python"]

    for k in ["version", "implementation", "packages"]:
        assert k in ctx

    assert list(ctx["packages"].keys()) == packages
    for v in ctx["packages"].values():
        assert v != ""
