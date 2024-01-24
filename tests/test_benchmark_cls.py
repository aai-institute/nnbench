import inspect

from nnbench import types


def test_interface_with_no_arguments():
    def empty_function() -> None:
        pass

    interface = types.Interface.from_callable(empty_function)
    assert interface.names == ()
    assert interface.types == ()
    assert interface.variables == ()
    assert interface.defaults == {}


def test_interface_with_multiple_arguments():
    def complex_function(a: int, b, c: str = "hello", d: float = 10.0) -> None:  # type:ignore
        pass

    interface = types.Interface.from_callable(complex_function)
    assert interface.names == ("a", "b", "c", "d")
    assert interface.types == (
        int,
        inspect._empty,
        str,
        float,
    )

    assert interface.variables == (("a", int), ("b", inspect._empty), ("c", str), ("d", float))
    assert interface.defaults == {"a": inspect._empty, "b": inspect._empty, "c": "hello", "d": 10.0}
