"""The ``nnbench`` command line interface."""

import argparse
import logging
import multiprocessing
import sys
import time
from collections.abc import Callable, Iterable
from functools import partial
from os import PathLike
from pathlib import Path
from typing import Any

from nnbench import __version__, collect, run
from nnbench.config import NNBenchConfig, import_, parse_nnbench_config
from nnbench.context import Context, ContextProvider
from nnbench.reporter import get_reporter_implementation
from nnbench.runner import jsonify
from nnbench.types import BenchmarkRecord
from nnbench.util import all_python_files

_VERSION = f"%(prog)s version {__version__}"
logger = logging.getLogger("nnbench")


class CustomFormatter(argparse.RawDescriptionHelpFormatter):
    def _format_action_invocation(self, action):
        if not action.option_strings:
            (metavar,) = self._metavar_formatter(action, action.dest)(1)
            return metavar
        elif isinstance(action, argparse.BooleanOptionalAction):
            if len(action.option_strings) == 2:
                true_opt, false_opt = action.option_strings
                return "--[no-]" + true_opt[2:]
        else:
            parts = []
            # if the Optional doesn't take a value, format is:
            #    -s, --long
            if action.nargs == 0:
                parts.extend(action.option_strings)

            # if the Optional takes a value, format is:
            #    -s, --long ARGS
            else:
                default = action.dest.upper()
                args_string = self._format_args(action, default)
                parts.extend(action.option_strings)
                parts[-1] += f" {args_string}"
            return ", ".join(parts)


def _log_level(log_level: str) -> str:
    """
    Initializes a stream handler to the nnbench logger if any level except NOTSET is passed.

    This runs before input validation against "choices", so we must check the input is in fact a
    literal of the available log levels.
    """

    if log_level not in ("NOTSET", "DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"):
        # ValueErrors are caught by argparse, but the error message is non-configurable,
        # and the original exception is swallowed
        raise ValueError

    if log_level != "NOTSET":
        logger.setLevel(log_level)
        sh = logging.StreamHandler()
        # TODO: Add --log-format switch to allow setting a custom log formatter (?)
        # TODO: Add --log-file switch to allow dumping to a file
        sh.setFormatter(
            logging.Formatter(fmt="[{levelname:<4} {name}:L{lineno}] {message}", style="{")
        )
        logger.addHandler(sh)
    return log_level


# hacky way to craft a nicer error message from a type hook:
# since __name__ is used as the argument value name, any weirdly
# named function (i.e. implementation detail) will produce a
# weird error message.
_log_level.__name__ = "log level"


def collect_and_run(
    path: str | PathLike[str],
    name: str | None = None,
    tags: tuple[str, ...] = (),
    context: Context | Iterable[ContextProvider] = (),
    jsonifier: Callable = jsonify,
) -> BenchmarkRecord:
    benchmarks = collect(path, tags=tags)
    record = run(
        benchmarks,
        name=name,
        context=context,
        jsonifier=jsonifier,
    )
    return record


def construct_parser(config: NNBenchConfig) -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser("nnbench", formatter_class=CustomFormatter)
    parser.add_argument("--version", action="version", version=_VERSION)
    parser.add_argument(
        "--log-level",
        default=config.log_level,
        type=_log_level,
        metavar="<level>",
        help="Log level to use for the nnbench package, defaults to NOTSET (no logging).",
    )
    subparsers = parser.add_subparsers(
        title="Available commands",
        required=False,
        dest="command",
        metavar="",
    )
    run_parser = subparsers.add_parser(
        "run", help="Run a benchmark workload.", formatter_class=CustomFormatter
    )
    # can be a directory, single file, or glob
    run_parser.add_argument(
        "benchmarks",
        nargs="?",
        metavar="<benchmarks>",
        default="benchmarks",
        help="A Python file or directory of files containing benchmarks to run.",
    )
    run_parser.add_argument(
        "-n",
        "--name",
        type=str,
        default=f"nnbench-{time.time_ns()}",
        metavar="<name>",
        help="A name to assign to the benchmark run, for example for record keeping in a database.",
    )
    run_parser.add_argument(
        "-j",
        type=int,
        default=-1,
        dest="jobs",
        metavar="<N>",
        help="Number of processes to use for running benchmarks in parallel, default: -1 (no parallelism)",
    )
    run_parser.add_argument(
        "--context",
        action="append",
        metavar="<key=value>",
        help="Additional context values giving information about the benchmark run.",
        default=list(),
    )
    run_parser.add_argument(
        "-t",
        "--tag",
        action="append",
        metavar="<tag>",
        dest="tags",
        help="Only run benchmarks marked with one or more given tag(s).",
        default=tuple(),
    )
    run_parser.add_argument(
        "-o",
        "--output-file",
        metavar="<file>",
        dest="outfile",
        help="File or stream to write results to, defaults to stdout.",
        default=sys.stdout,
    )
    run_parser.add_argument(
        "--jsonifier",
        metavar="<classpath>",
        default=config.jsonifier,
        help="Function to create a JSON representation of input parameters with, helping make runs reproducible.",
    )

    compare_parser = subparsers.add_parser(
        "compare",
        help="Compare results from multiple benchmark runs.",
        formatter_class=CustomFormatter,
    )
    compare_parser.add_argument(
        "records",
        nargs="+",
        help="Records to compare results for. Can be given as local files or remote URIs.",
    )
    compare_parser.add_argument(
        "-P",
        "--include-parameter",
        action="append",
        metavar="<name>",
        dest="parameters",
        default=list(),
        help="Names of input parameters to display in the comparison table.",
    )
    compare_parser.add_argument(
        "-C",
        "--include-context",
        action="append",
        metavar="<name>",
        dest="contextvals",
        default=list(),
        help="Context values to display in the comparison table. Use dotted syntax for nested context values.",
    )
    compare_parser.add_argument(
        "-E",
        "--extra-column",
        action="append",
        metavar="<name>",
        dest="extra_cols",
        default=list(),
        help="Additional record data to display in the comparison table.",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    """The main ``nnbench`` CLI entry point."""
    config = parse_nnbench_config()
    parser = construct_parser(config)
    try:
        args = parser.parse_args(argv)
        if args.command is None:
            parser.print_help()
            return 1
        elif args.command == "run":
            from nnbench.context import builtin_providers, register_context_provider

            context: dict[str, Any] = {}
            for p in config.context:
                klass = import_(p.classpath)
                # TODO: Move registration to nnbench config parsing
                register_context_provider(p.name, klass, p.arguments)

            for val in args.context:
                if val in builtin_providers:
                    context.update(builtin_providers[val]())
                else:
                    try:
                        k, v = val.split("=", 1)
                        context[k] = v
                    except ValueError:
                        raise ValueError("context values need to be of the form <key>=<value>")

            n_jobs: int = args.jobs
            jsonifier = import_(args.jsonifier)
            if n_jobs < 2:
                record = collect_and_run(
                    args.benchmarks,
                    name=args.name,
                    tags=tuple(args.tags),
                    context=context,
                    jsonifier=jsonifier,
                )
            else:
                compute_fn = partial(
                    collect_and_run,
                    name=args.name,
                    tags=tuple(args.tags),
                    context=context,
                    jsonifier=jsonifier,
                )
                with multiprocessing.Pool(n_jobs) as p:
                    bm_path = Path(args.benchmarks)
                    # unroll paths in case a directory is passed.
                    if bm_path.is_dir():
                        bm_files = all_python_files(bm_path)
                    else:
                        bm_files = [bm_path]
                    res: list[BenchmarkRecord] = p.map(compute_fn, bm_files)
                    start: list[dict[str, Any]] = list()
                    benchmarks = sum([r.benchmarks for r in res], start=start)
                    # Assumes the context and run name to be consistent across workers.
                    record = BenchmarkRecord(
                        run=res[0].run,
                        context=res[0].context,
                        benchmarks=benchmarks,
                    )

            outfile = args.outfile
            reporter = get_reporter_implementation(outfile)
            reporter.write(record, outfile, {})
        elif args.command == "compare":
            from nnbench.compare import Comparison

            records: list[BenchmarkRecord] = []
            for file in args.records:
                reporter = get_reporter_implementation(file)
                records.append(reporter.read(file, {}))  # TODO: Support options
            Comparison().render(records)

        return 0
    except Exception as e:
        sys.stderr.write(f"error: {e}")
        return 1
