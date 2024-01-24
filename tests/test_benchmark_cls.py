import inspect
import nnbench


def test_interface_with_no_arguments():
    def empty_function() -> None:
        pass

    interface = nnbench.Benchmark.Interface(empty_function)
    assert interface.varnames == ()
    assert interface.vartypes == ()
    assert interface.varitems == ()
    assert interface.defaults == {}


def test_interface_with_multiple_arguments():
    def complex_function(a: int, b, c: str = "hello", d: float = 10.0) -> None:  # type:ignore
        pass

    interface = nnbench.Benchmark.Interface(complex_function)
    assert interface.varnames == ("a", "b", "c", "d")
    assert tuple(param.annotation for param in interface.vartypes) == (
        int, inspect._empty, str, float)

    varitems = [(param.name, param.annotation) for name, param in interface.varitems]
    assert varitems == [
        ("a", int),
        ("b", inspect._empty),
        ("c", str),
        ("d", float)
    ]
    assert interface.defaults == {
        'a': inspect._empty, 'b': inspect._empty, 'c': 'hello', 'd': 10.0}
