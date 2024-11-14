import argparse
import sys
from typing import Any

from nnbench import default_reporter, default_runner
from nnbench.reporter.file import FileIO


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
        metavar="<key>=<value>",
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

    args = parser.parse_args()

    runner = default_runner()
    reporter = default_reporter()

    context: dict[str, Any] = {}
    for val in args.context:
        try:
            k, v = val.split("=")
        except ValueError:
            raise ValueError("context values need to be of the form <key>=<value>")
        context[k] = v

    record = runner.run(args.benchmarks, tags=tuple(args.tags))

    outfile = args.outfile
    if args.outfile == sys.stdout:
        reporter.display(record)
    else:
        f = FileIO()
        f.write(record, outfile)

    return 0
