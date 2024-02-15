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


def test_unflatten_dictionary():
    flat_ctx = Context(data={"a": 1, "b.c": 2, "b.d.e": 3})
    unflattened = flat_ctx.unflatten(inplace=False)
    assert unflattened._ctx_dict == {"a": 1, "b": {"c": 2, "d": {"e": 3}}}


def test_context_keys():
    ctx = Context({"a": 1, "b": {"c": 2}})
    expected_keys = set(["a", "b.c"])
    assert set(ctx.keys()) == expected_keys


def test_context_values():
    ctx = Context({"a": 1, "b": {"c": 2}})
    expected_values = set([1, 2])
    assert set(ctx.values()) == expected_values


def test_context_items():
    ctx = Context({"a": 1, "b": {"c": 2}})
    expected_items = set([("a", 1), ("b.c", 2)])
    assert set(ctx.items()) == expected_items


def test_update_with_unnested_dict():
    ctx = Context({"a": 1, "b": 2})
    ctx.update({"a": 3, "c": 4})
    expected_dict = {"a": 3, "b": 2, "c": 4}
    assert ctx._ctx_dict == expected_dict


def test_update_with_nested_dict():
    ctx = Context({"a": 1})
    ctx.update({"b": {"c": 2}})
    expected_dict = {"a": 1, "b": {"c": 2}}
    assert ctx._ctx_dict == expected_dict


def test_update_unnested_with_nested_dict():
    ctx = Context({"a": 1, "b": 2})
    ctx.update({"b": {"c": 3}})
    expected_dict = {"a": 1, "b": {"c": 3}}
    assert ctx._ctx_dict == expected_dict


def test_update_nested_with_unnested_dict():
    ctx = Context({"a": {"b": 2}})
    ctx.update({"a": 3, "c": 4})
    expected_dict = {"a": 3, "c": 4}
    assert ctx._ctx_dict == expected_dict


def test_update_with_context_instance():
    ctx1 = Context({"a": 1, "b": {"c": 2}})
    ctx2 = Context({"d": 4})
    ctx1.update(ctx2)
    expected_dict = {"a": 1, "b": {"c": 2}, "d": 4}
    assert ctx1._ctx_dict == expected_dict
