"""The ``nnbench`` command line interface."""

import argparse
import importlib
import logging
import sys
from typing import Any

from nnbench import ConsoleReporter, __version__, collect, run
from nnbench.config import NNBenchConfig, parse_nnbench_config
from nnbench.reporter import FileReporter

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
        help="Python file or directory of files containing benchmarks to run.",
        default="benchmarks",
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


def main() -> int:
    """The main ``nnbench`` CLI entry point."""
    config = parse_nnbench_config()
    parser = construct_parser(config)
    try:
        args = parser.parse_args()
        if args.command is None:
            parser.print_help()
            return 1
        elif args.command == "run":
            from nnbench.context import builtin_providers

            context: dict[str, Any] = {}
            for p in config.context:
                modname, classname = p.classpath.rsplit(".", 1)

                # TODO: (n.junge) Avoid f-string interpolation in logs
                logger.debug(f"Registering context provider {p.name!r}")

                # TODO: Catch import errors if the module does not exist
                klass = getattr(importlib.import_module(modname), classname)
                if isinstance(klass, type):
                    # classes can be instantiated with arguments,
                    # while functions cannot.
                    builtin_providers[p.name] = klass(**p.arguments)
                else:
                    builtin_providers[p.name] = klass

            for val in args.context:
                try:
                    k, v = val.split("=", 1)
                except ValueError:
                    raise ValueError("context values need to be of the form <key>=<value>")
                if k == "provider":
                    try:
                        context.update(builtin_providers[v]())
                    except KeyError:
                        raise KeyError(f"unknown context provider {v!r}") from None
                else:
                    context[k] = v

            benchmarks = collect(args.benchmarks, tags=tuple(args.tags))
            record = run(
                benchmarks,
                context=[lambda: context],
            )

            outfile = args.outfile
            if outfile == sys.stdout:
                reporter = ConsoleReporter()
                reporter.display(record)
            else:
                f = FileReporter()
                f.write(record, outfile)
        elif args.command == "compare":
            from nnbench.compare import compare

            f = FileReporter()
            records = [f.read(file) for file in args.records]
            compare(
                records=records,
                parameters=args.parameters,
                contextvals=args.contextvals,
            )

        return 0
    except Exception as e:
        sys.stderr.write(f"error: {e}")
        sys.exit(1)
