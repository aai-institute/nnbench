import inspect
import os
from collections.abc import Callable, Iterable
from pathlib import Path
from types import ModuleType
from typing import Any

from nnbench.types import Benchmark, Interface
from nnbench.util import import_file_as_module


def get_transitive_closure(mod: ModuleType, name: str) -> tuple[list[Callable], list[Interface]]:
    fixture = getattr(mod, name)
    if not callable(fixture):
        raise ValueError(f"fixture input {name!r} needs to be a callable")
    closure: list[Callable] = [fixture]
    interfaces: list[Interface] = []

    def recursive_closure_collection(fn, _closure):
        _if = Interface.from_callable(fn, {})
        interfaces.append(_if)
        # if the fixture itself takes arguments,
        # resolve all of them within the module.
        for closure_name in _if.names:
            _closure_obj = getattr(mod, closure_name, None)
            if _closure_obj is None:
                raise ImportError(f"fixture {name!r}: missing closure value {closure_name!r}")
            if not callable(_closure_obj):
                raise ValueError(f"input {name!r} to fixture {fn} needs to be a callable")
            _closure.append(_closure_obj)
            recursive_closure_collection(_closure_obj, _closure)

    recursive_closure_collection(fixture, closure)
    return closure, interfaces


class FixtureManager:
    """
    A lean class responsible for resolving parameter values (aka 'fixtures')
    of benchmarks from provider functions.

    To resolve a benchmark parameter (in resolve()), the class does
    the following:
        1. Obtain the path to the file containing the benchmark, as
         the __file__ attribute of the benchmark function's origin module.
        2. Look for a `conf.py` file in the same directory.
        3. Import the `conf.py` module, look for a function named the same as
         the benchmark parameter.
        4. If necessary, resolve any named inputs to the function **within**
         the module scope.
        5. If no function member is found, and the benchmark file is not in `root`,
         fall back to the parent directory, repeat steps 2-5, until `root` is reached.
        6. If no `conf.py` contains any function matching the name, throw an
         error (TODO: ImportError? custom?)
    """

    def __init__(self, root: str | os.PathLike[str]) -> None:
        self.root = Path(root)
        self.cache: dict[Path, dict[str, Any]] = {}
        """
        Cache architecture:
        key: directory
        value: key-value mapping of fixture name -> fixture value within directory.
        """

    def collect(self, mod: ModuleType, names: Iterable[str]) -> dict[str, Any]:
        """
        Given a fixture module and a list of fixture names required (for a
        selected benchmark), collect values, computing transitive closures in the
        meantime (i.e., inputs required to compute certain fixtures), and add
        the resulting values to the cache.
        """
        res: dict[str, Any] = {}
        for name in names:
            fixture_cand = getattr(mod, name, None)
            if fixture_cand is None:
                continue
            else:
                closure, interfaces = get_transitive_closure(mod, name)
                # easy case first, fixture without arguments - just call the function.
                if len(closure) == 1:
                    (fn,) = closure
                    res[name] = fn()
                else:
                    # compute the closure in reverse to get the fixture value.
                    # the last fixture takes no arguments, otherwise this would
                    # be an infinite loop.
                    idx = -1
                    _temp_res: dict[str, Any] = {}
                    for iface in reversed(interfaces):
                        iface_names = iface.names
                        # each fixture can take values in the respective closure,
                        # but only values that have already been computed.
                        kwargs = {k: v for k, v in _temp_res.items() if k in iface_names}
                        fn = closure[idx]
                        _temp_res[iface.funcname] = fn(**kwargs)
                        idx -= 1
                    assert name in _temp_res, f"internal error computing fixture {name!r}"
                    res[name] = _temp_res[name]
        return res

    def resolve(self, bm: Benchmark) -> dict[str, Any]:
        fixturevals: dict[str, Any] = {}
        # first, get the candidate fixture names, aka the benchmark param names.
        # We limit ourselves to names that do not have a default value.
        names = [
            n
            for n, d in zip(bm.interface.names, bm.interface.defaults)
            if d == inspect.Parameter.empty
        ]
        nameset, fixtureset = set(names), set()
        # then, load the benchmark function's origin module,
        # should be fast as it's a sys.modules lookup.
        # Each user module loaded via spec_from_file_location *should* have its
        # __file__ attribute set, so inspect.getsourcefile can find it.
        bm_origin_module = inspect.getmodule(bm.fn)
        sourcefile = inspect.getsourcefile(bm_origin_module)

        if sourcefile is None:
            raise ValueError(
                "during fixture collection: "
                f"could not locate origin module for benchmark {bm.name!r}()"
            )

        # then, look for a `conf.py` file in the benchmark file's directory,
        # and all parents up to "root" (i.e., the path in `nnbench run <path>`)
        bm_file = Path(sourcefile)
        for p in bm_file.parents:
            conf_candidate = p / "conf.py"
            if conf_candidate.exists():
                bm_dir_cache: dict[str, Any] | None = self.cache.get(p, None)
                if bm_dir_cache is None:
                    mod = import_file_as_module(conf_candidate)
                    # contains fixture values for the benchmark that could be resolved
                    # on the current directory level.
                    res = self.collect(mod, names)
                    # hydrate the directory cache with the found fixture values.
                    # some might be missing, so we could have to continue traversal.
                    self.cache[p] = res
                    fixturevals.update(res)
                else:
                    # at this point, the cache entry might have other fixture values
                    # that this benchmark may not consume, so we need to filter.
                    fixturevals.update({k: v for k, v in bm_dir_cache.items() if k in names})

                fixtureset |= set(fixturevals)

            if p == self.root or nameset == fixtureset:
                break

        # TODO: This should not throw an error, inline the typecheck into before benchmark
        # execution, then handle it there.
        if fixtureset < nameset:
            missing, *_ = nameset - fixtureset
            raise RuntimeError(f"could not locate fixture {missing!r} for benchmark {bm.name!r}")

        return fixturevals
