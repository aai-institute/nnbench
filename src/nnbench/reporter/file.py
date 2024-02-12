from __future__ import annotations

import os
from typing import Any, List

from nnbench.reporter.base import BenchmarkReporter
from nnbench.types import BenchmarkRecord


class Parser:
    """The base interface for parsing records form file.

    Usage:
    ------
    ```
    class MyCustomParser(Parser):
        def parse_file(self, records):
            # Implement your custom parsing logic here
            ...
        def write_records(self, records):
            # Implement your custom file writing logic here
            ...
    # Register your custom parser with a distinct file type
    MyCustomParser.register("my_custom_format")
    # Usage:
    new_record = ...  # Load records in your custom format
    append_record_to_records(records, new_record, "my_custom_format")
    ```
    """

    def parse_file(self, records: str) -> Any:
        """Parses records and returns a list of parsed data.

        Args:
        -----
            `records:` A list or iterator of record strings.

        Returns:
        --------
            A list of parsed records.
        """
        raise NotImplementedError

    def write_records(self, records: Any[BenchmarkRecord], record: BenchmarkRecord) -> str:
        """Appends a record to the existing records based on the file type.

        Args:
        -----
            `records:` A list of parsed records.
            `record:` The record string to append.
            `file_type:` The file type (string).

        Returns:
        --------
            A string form of the content to be written in a file.
        """
        raise NotImplementedError

    @classmethod
    def register(cls, file_type: str) -> None:
        """Registers a parser for a specific file type.

        Args:
            `file_type:` The file type (string)
        """
        parsers[file_type] = cls

    @staticmethod
    def get_parser(file_type: str):
        """Gets the registered parser for a file type.

        Args:
            `file_type:` The file type (string)

        Returns:
        --------
            The registered RecordParser, or None if not found.
        """
        return parsers.get(file_type)


class JsonParser(Parser):
    def parse_file(self, records: str) -> List[dict]:
        import json

        try:
            return json.loads(records if records else "[]")
        except json.JSONDecodeError:
            raise ValueError("Unexpected records passed")

    def write_records(
        self, parsed_records: Any[BenchmarkRecord] | None, record: BenchmarkRecord
    ) -> str:
        import json

        parsed_records.append(record)
        return json.dumps(parsed_records)


class YamlParser(Parser):
    def parse_file(self, records: str) -> List[dict]:
        import yaml

        return yaml.safe_load(records) if records else []

    def write_records(
        self, parsed_records: Any[BenchmarkRecord] | None, record: BenchmarkRecord
    ) -> str:
        import yaml

        parsed_records.append(record)
        for element in record["benchmarks"]:
            element["value"] = float(element["value"])
        return yaml.dump(parsed_records)


# Register custom parsers here
parsers = {"json": JsonParser, "yaml": YamlParser}


def parse_records(records: str, file_type: str) -> Any:
    """Parses records based on the specified file type.

    This function retrieves and calls the registered parser for
    the given file type.

    Args:
        `records:` A list or iterator of record strings.
        `file_type:` The file type (string).

    Returns:
        A list of parsed records.
    """

    parser = Parser.get_parser(file_type)
    if parser is None:
        raise ValueError(f"Unsupported file type: {file_type}")

    return parser().parse_file(records)


def append_record_to_records(parsed_records: Any, record: BenchmarkRecord, file_type: str) -> str:
    """Appends a record to the list based on the file type.

    This function first parses the record using the appropriate parser
    and then appends it to the `parsed_records`.

    Args:
        `records:` A list of parsed records.
        `record:` The record to append.
        `file_type:` The file type (string).
    """

    parser = Parser.get_parser(file_type)
    if parser is None:
        raise ValueError(f"Unsupported file type: {file_type}")

    return parser().write_records(parsed_records, record)


class FileReporter(BenchmarkReporter):
    def __init__(self, dir: str):
        self.dir = dir
        if not os.path.exists(dir):
            self.initialize()

    def initialize(self) -> None:
        try:
            os.makedirs(self.dir, exist_ok=True)
        except OSError as e:
            self.finalize()
            raise ValueError(f"Could not create directory: {self.dir}") from e

    def read(self, file_name: str) -> BenchmarkRecord:
        if not self.dir:
            raise BaseException("Directory is not initialized")
        file_path = os.path.join(self.dir, file_name)
        file_type = file_name.split(".")[1]
        try:
            with open(file_path) as file:
                data = file.read()
                parsed_data = parse_records(data, file_type)
                return parsed_data
        except FileNotFoundError:
            raise ValueError(f"Could not read the file: {file_path}")

    def write(self, record: BenchmarkRecord, file_name: str) -> None:
        if not self.dir:
            raise BaseException("Directory is not initialized")

        file_path = os.path.join(self.dir, file_name)
        if not os.path.exists(file_path):  # Create the file
            with open(file_path, "w") as file:
                file.write("")
        try:
            parsed_records = self.read(file_name)
            file_type = file_name.split(".")[1]
            new_records = append_record_to_records(parsed_records, record, file_type)
            with open(file_path, "w") as file:
                file.write(new_records)
        except FileNotFoundError:
            raise ValueError(f"Could not read the file: {file_path}")

    def finalize(self) -> None:
        del self.dir
