class BenchmarkReporter:
    """
    The base interface for a benchmark reporter class.

    A benchmark reporter consumes benchmark results from a previous run, and subsequently
    reports them in the way specified by the respective implementation's ``report_result()``
    method.

    For example, to write benchmark results to a database, you could save the credentials
    for authentication on the class, and then stream the results directly to
    the database in ``report_result()``, with preprocessing if necessary.
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._initialized = False

    def initialize(self):
        """
        Initialize the reporter's state.

        This is the intended place to create resources like a result directory,
        a database connection, or a HTTP client.
        """
        self._initialized = True

    def finalize(self):
        """
        Finalize the reporter's state.

        This is the intended place to destroy/release resources that were previously
        acquired in ``initialize()``.
        """
        pass
