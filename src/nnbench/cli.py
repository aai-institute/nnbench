import argparse
import sys
from typing import Any

from nnbench import BenchmarkReporter, BenchmarkRunner
from nnbench.reporter.file import FileReporter


def main() -> int:
    parser = argparse.ArgumentParser("nnbench")
    # can be a directory, single file, or glob
    parser.add_argument(
        "benchmarks",
        nargs="?",
        metavar="<benchmarks>",
        help="Python file or directory of files containing benchmarks to run.",
        default="benchmarks",
    )
    parser.add_argument(
        "--context",
        action="append",
        metavar="<key=value>",
        help="Additional context values giving information about the benchmark run.",
        default=list(),
    )
    parser.add_argument(
        "-t",
        "--tag",
        action="append",
        metavar="<tag>",
        dest="tags",
        help="Only run benchmarks marked with one or more given tag(s).",
        default=tuple(),
    )
    parser.add_argument(
        "-o",
        "--output-file",
        metavar="<file>",
        dest="outfile",
        help="File or stream to write results to.",
        default=sys.stdout,
    )
    parser.add_argument(
        "--typecheck",
        action=argparse.BooleanOptionalAction,
        help="Whether or not to strictly check types of benchmark inputs.",
    )

    args = parser.parse_args()

    context: dict[str, Any] = {}
    for val in args.context:
        try:
            k, v = val.split("=")
        except ValueError:
            raise ValueError("context values need to be of the form <key>=<value>")
        context[k] = v

    record = BenchmarkRunner().run(args.benchmarks, tags=tuple(args.tags))

    outfile = args.outfile
    if args.outfile == sys.stdout:
        reporter = BenchmarkReporter()
        reporter.display(record)
    else:
        f = FileReporter()
        f.write(record, outfile)

    return 0
