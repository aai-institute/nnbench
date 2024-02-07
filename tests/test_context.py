from nnbench.context import CPUInfo, GitEnvironmentInfo, PythonInfo


def test_python_package_info() -> None:
    p = PythonInfo("pre-commit", "pyyaml")()
    res = p["python"]

    deps = res["dependencies"]
    for v in deps.values():
        assert v != ""

    # for a bogus package, it should not fail but produce an empty string.
    p = PythonInfo("asdfghjkl")()
    res = p["python"]

    deps = res["dependencies"]
    for v in deps.values():
        assert v == ""


def test_git_info_provider() -> None:
    g = GitEnvironmentInfo()()
    res = g["git"]
    # tag might not be available in a shallow checkout in CI,
    # but commit, provider and repo are.
    assert res["commit"] != ""
    assert res["provider"] == "github.com"
    assert res["repository"] == "aai-institute/nnbench"


def test_cpu_info_provider() -> None:
    c = CPUInfo()()
    res = c["cpu"]

    # just check that the most important fields are populated across
    # the popular CPU architectures.
    assert res["architecture"] != ""
    assert res["system"] != ""
    assert res["frequency"] > 0
    assert res["num_cpus"] > 0
    assert res["total_memory"] > 0
