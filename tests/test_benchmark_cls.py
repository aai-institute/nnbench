import inspect

from nnbench import types


def test_interface_with_no_arguments():
    def fn() -> None:
        pass

    interface = types.Interface.from_callable(fn)
    assert interface.names == ()
    assert interface.types == ()
    assert interface.defaults == ()
    assert interface.variables == ()
    assert interface.returntype is type(None)


def test_interface_with_multiple_arguments():
    def fn(a: int, b, c: str = "hello", d: float = 10.0) -> None:  # type: ignore
        pass

    interface = types.Interface.from_callable(fn)
    empty = inspect.Parameter.empty
    assert interface.names == ("a", "b", "c", "d")
    assert interface.types == (int, empty, str, float)
    assert interface.defaults == (empty, empty, "hello", 10.0)
    assert interface.variables == (
        ("a", int, empty),
        ("b", empty, empty),
        ("c", str, "hello"),
        ("d", float, 10.0),
    )
    assert interface.returntype is type(None)
