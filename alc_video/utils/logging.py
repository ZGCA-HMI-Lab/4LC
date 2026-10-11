from __future__ import annotations

import logging
import os
import sys
import time
from contextlib import contextmanager
from pathlib import Path

LOG = logging.getLogger("fourlc")
from alc_video.utils.errors import RequestError

LOG_LEVELS = ("DEBUG", "INFO", "WARNING", "ERROR")


class _RankFilter(logging.Filter):
    def __init__(self, rank: int):
        super().__init__()
        self.rank = rank

    def filter(self, record: logging.LogRecord) -> bool:
        record.rank = self.rank
        return True


def configure_logging(
    level: str = "INFO", *, path: Path | None = None, rank: int | None = None
) -> None:
    rank = int(os.environ.get("RANK", "0")) if rank is None else rank
    selected = getattr(logging, level)
    selected = selected if rank == 0 else max(selected, logging.WARNING)
    for handler in LOG.handlers[:]:
        LOG.removeHandler(handler)
        handler.close()
    LOG.setLevel(selected)
    LOG.propagate = False
    formatter = logging.Formatter(
        "%(asctime)s %(levelname)s rank=%(rank)s %(message)s", datefmt="%H:%M:%S"
    )
    handlers: list[logging.Handler] = [logging.StreamHandler(sys.stderr)]
    if path is not None:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        if rank:
            path = path.with_name(f"{path.stem}.rank{rank}{path.suffix}")
        handlers.append(logging.FileHandler(path, encoding="utf-8", delay=True))
    for handler in handlers:
        handler.setFormatter(formatter)
        handler.addFilter(_RankFilter(rank))
        LOG.addHandler(handler)
    library = logging.getLogger("diffsynth")
    library.handlers = LOG.handlers[:]
    library.propagate = False
    quiet_libraries()


def quiet_libraries() -> None:
    level = logging.DEBUG if LOG.isEnabledFor(logging.DEBUG) else logging.WARNING
    for name in (
        "DeepSpeed",
        "deepspeed",
        "accelerate",
        "transformers",
        "diffusers",
        "diffsynth",
    ):
        logging.getLogger(name).setLevel(level)


def _details(fields: dict) -> str:
    return " ".join(f"{name}={value}" for name, value in fields.items())


@contextmanager
def stage(name: str, *, level: int = logging.INFO, **context):
    started = time.perf_counter()
    details = _details(context)
    LOG.log(level, "%s start %s", name, details)
    try:
        yield
    except BaseException:
        LOG.error(
            "%s failed elapsed=%.2fs %s", name, time.perf_counter() - started, details
        )
        raise
    else:
        LOG.log(
            level,
            "%s complete elapsed=%.2fs %s",
            name,
            time.perf_counter() - started,
            details,
        )


def run_cli(operation) -> None:
    try:
        operation()
    except RequestError as error:
        LOG.error("%s", error)
        raise SystemExit(2) from None
    except Exception:
        LOG.exception("run failed")
        raise SystemExit(1) from None
