import pytest

from nnbench.context import Context, CPUInfo, GitEnvironmentInfo, PythonInfo

# Sample context dicts for testing
nested_dict = {"a": 1, "b": {"c": 2, "d": {"e": 3}}}

flat_dict = {"a": 1, "b.c": 2, "b.d.e": 3}


@pytest.fixture
def context_instance():
    return Context()


@pytest.fixture
def flat_context_instance():
    return Context(flat_dict)


@pytest.fixture
def nested_context_instance():
    return Context(nested_dict)


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


def test_merge_flat_dictionaries(context_instance):
    context_instance.merge({"a": 1}, inplace=True)
    context_instance.merge({"b": 2}, inplace=True)
    assert context_instance._ctx_dict == {"a": 1, "b": 2}


def test_merge_callable(context_instance):
    context_instance.merge(lambda: {"a": 1}, inplace=True)
    assert context_instance._ctx_dict == {"a": 1}


def test_flatten_nested_dictionary(nested_context_instance):
    flattened = nested_context_instance.flatten(inplace=False)
    assert flattened._ctx_dict == flat_dict


def test_flatten_with_prefix(nested_context_instance):
    flattened = nested_context_instance.flatten(prefix="prefix", inplace=False)
    expected_dict = {f"prefix.{k}": v for k, v in flat_dict.items()}
    assert flattened._ctx_dict == expected_dict


def test_unflatten_dictionary(flat_context_instance):
    unflattened = flat_context_instance.unflatten(inplace=False)
    assert unflattened._ctx_dict == nested_dict


def test_filter_flat_dictionary():
    context = Context({"a": 1, "b": "test", "c": 2})
    filtered = context.filter(lambda k, v: isinstance(v, int), inplace=False)
    assert filtered._ctx_dict == {"a": 1, "c": 2}


def test_filter_nested_dictionary():
    context = Context(nested_dict)
    filtered = context.filter(lambda k, v: v > 1, inplace=False)
    expected_dict = {"b": {"c": 2, "d": {"e": 3}}}
    assert filtered._ctx_dict == expected_dict
