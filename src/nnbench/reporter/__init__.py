"""
An interface for displaying, writing, or streaming benchmark results to
files, databases, or web services.
"""

from .base import BenchmarkReporter
from .console import ConsoleReporter
from .duckdb_sql import DuckDBReporter
from .file import FileReporter
