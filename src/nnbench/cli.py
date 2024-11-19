import argparse
import sys
from typing import Any

from nnbench import BenchmarkRunner, ConsoleReporter, __version__
from nnbench.reporter.file import FileReporter

_VERSION = f"%(prog)s version {__version__}"


def main() -> int:
    parser = argparse.ArgumentParser("nnbench")
    parser.add_argument("--version", action="version", version=_VERSION)
    subparsers = parser.add_subparsers(
        metavar="<command>", required=False, dest="command", help="Subcommands"
    )
    run_parser = subparsers.add_parser("run", help="Run a benchmark workload.")
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
    run_parser.add_argument(
        "--typecheck",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Whether or not to strictly check types of benchmark inputs.",
    )

    compare_parser = subparsers.add_parser(
        "compare", help="Compare results from multiple benchmark runs."
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
    # TODO: Add customization option for rich table displays

    try:
        args = parser.parse_args()
        if args.command == "run":
            context: dict[str, Any] = {}
            for val in args.context:
                try:
                    k, v = val.split("=")
                except ValueError:
                    raise ValueError("context values need to be of the form <key>=<value>")
                # TODO: Support builtin providers in the runner
                context[k] = v

            record = BenchmarkRunner(typecheck=args.typecheck).run(
                args.benchmarks,
                tags=tuple(args.tags),
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
