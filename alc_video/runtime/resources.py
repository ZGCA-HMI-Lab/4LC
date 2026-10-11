from collections.abc import Callable, Iterator
from contextlib import contextmanager
from typing import Any

from alc_video.utils.logging import LOG


@contextmanager
def release_resource(cleanup: Callable[..., Any], *args: Any) -> Iterator[None]:
    try:
        yield
    except BaseException:
        try:
            cleanup(*args)
        except Exception:
            LOG.exception("resource cleanup failed while handling an execution error")
        raise
    else:
        cleanup(*args)
