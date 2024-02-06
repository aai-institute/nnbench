from nnbench.context import CPUInfo, GitEnvironmentInfo, PythonPackageInfo


def test_python_package_info() -> None:
    p = PythonPackageInfo("pre-commit", "pyyaml")()

    for v in p.values():
        assert v != ""

    # for a bogus package, it should not fail but produce an empty string.
    p = PythonPackageInfo("asdfghjkl")()

    for v in p.values():
        assert v == ""


def test_git_info_provider() -> None:
    g = GitEnvironmentInfo()()
    # tag might not be available in a shallow checkout in CI,
    # but commit, provider and repo are.
    assert g["commit"] != ""
    assert g["provider"] == "github.com"
    assert g["repository"] == "aai-institute/nnbench"


def test_cpu_info_provider() -> None:
    c = CPUInfo()()

    # just check that the most important fields are populated across
    # the popular CPU architectures.
    assert c["architecture"] != ""
    assert c["system"] != ""
    assert c["frequency"] > 0
    assert c["num_cpus"] > 0
    assert c["total_memory"] > 0
