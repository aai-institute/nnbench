from __future__ import annotations

import os

from nnbench.reporter.base import BenchmarkReporter
from nnbench.types import BenchmarkRecord

from abc import ABC, abstractmethod


class Parser(ABC):
    """Abstract base class for parsing records form file.
    
    Usage:
    ------
    ```
    class MyCustomParser(Parser):
        def parse(self, records):
            # Implement your custom parsing logic here
            ...
    # Register your custom parser with a distinct file type
    MyCustomParser.register("my_custom_format")
    # Usage:
    custom_records = ...  # Load records in your custom format
    append_record_to_records(records, custom_record, "my_custom_format")
    ```
    """

    @abstractmethod
    def parse(self, records):
        """Parses records and returns a list of parsed data.

        Args:
            records: A list or iterator of record strings.

        Returns:
            A list of parsed records.
        """

    @classmethod
    def register(cls, file_type):
        """Registers a parser for a specific file type.

        Args:
            file_type: The file type (string)
        """
        parsers[file_type] = cls

    @staticmethod
    def get_parser(file_type):
        """Gets the registered parser for a file type.

        Args:
            file_type: The file type (string)

        Returns:
            The registered RecordParser, or None if not found.
        """


class JsonParser(Parser):
    def parse(self, records):
        import json
        return [json.loads(record) for record in records]


class YamlParser(Parser):
    def parse(self, records):
        import yaml
        return [yaml.safe_load(record) for record in records]


parsers = {"json": JsonParser, "yaml": YamlParser}


def parse_records(records, file_type):
    """Parses records based on the specified file type.

    This function retrieves and calls the registered parser for
    the given file type.

    Args:
        records: A list or iterator of record strings.
        file_type: The file type (string).

    Returns:
        A list of parsed records.
    """

    parser = Parser.get_parser(file_type)
    if parser is None:
        raise ValueError(f"Unsupported file type: {file_type}")

    return parser().parse(records)


def append_record_to_records(records, record, file_type):
    """Appends a record to the list based on the file type.

    This function first parses the record using the appropriate parser
    and then appends it to the `records` list.

    Args:
        records: A list of parsed records.
        record: The record string to append.
        file_type: The file type (string).
    """

    parsed_record = parse_records([record], file_type)[0]
    records.append(parsed_record)


class FileReporter(BenchmarkReporter):
    def __init__(self, dir: str):
        if not os.path.exists(dir):
            self.initialize(dir)
        self.dir = dir

    def initialize(self, path: str) -> None:
        try:
            os.makedirs(path, exist_ok=True)
        except OSError as e:
            raise ValueError(f"Could not create directory: {path}") from e
                
    def read(self, file_name: str) -> BenchmarkRecord:
        if not self.dir:
            raise (f"Directory is not initialized")
        file_path = os.path.join(self.dir, file_name)
        file_type = file_name.split('.')[1]
        try:
            with open(file_path) as file:
                data = file.read()
                parsed_data = parse_records(data, file_type)
                return parsed_data
        except:
            raise ValueError(f"Could not read the file: {file_path}")
    
    def write(self, file_name, record: BenchmarkRecord) -> None:
        if not self.dir:
            raise (f"Directory is not initialized")
        
        file_path = os.path.join(self.dir, file_name)
        try:
            records = self.read(file_name)
            file_type = file_name.split('.')[1]
            new_records = append_record_to_records(records, record, file_type)
            with open(file_path, 'w') as file:
                file.write(new_records)
        except:
            raise ValueError(f"Could not read the file: {file_path}")

    def finalize(self) -> None:
        del self.dir
