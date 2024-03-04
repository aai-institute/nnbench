"""
A lightweight interface for refining, displaying, and streaming benchmark results to various sinks.
"""

from __future__ import annotations

from .base import BenchmarkReporter
from .duckdb_sql import DuckDBReporter
from .file import FileReporter
