from nnbench.context import Context, CPUInfo, GitEnvironmentInfo, PythonInfo


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


def test_flatten_nested_dictionary():
    nested_ctx = Context({"a": 1, "b": {"c": 2, "d": {"e": 3}}})
    flattened = nested_ctx.flatten(inplace=False)
    assert flattened._ctx_dict == {"a": 1, "b.c": 2, "b.d.e": 3}


def test_flatten_with_prefix():
    nested_ctx = Context({"a": 1, "b": {"c": 2, "d": {"e": 3}}})
    flattened = nested_ctx.flatten(prefix="prefix", inplace=False)
    expected_dict = {f"prefix.{k}": v for k, v in {"a": 1, "b.c": 2, "b.d.e": 3}.items()}
    assert flattened._ctx_dict == expected_dict


def test_unflatten_dictionary():
    flat_ctx = Context(data={"a": 1, "b.c": 2, "b.d.e": 3})
    unflattened = flat_ctx.unflatten(inplace=False)
    assert unflattened._ctx_dict == {"a": 1, "b": {"c": 2, "d": {"e": 3}}}
