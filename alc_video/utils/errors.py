from collections.abc import Iterator
from contextlib import contextmanager


class RequestError(ValueError):
    """An input or configuration error the caller can correct."""


@contextmanager
def input_errors() -> Iterator[None]:
    try:
        yield
    except RequestError:
        raise
    except (ValueError, TypeError, LookupError, OSError) as error:
        raise RequestError(str(error)) from error
