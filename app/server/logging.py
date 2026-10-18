from __future__ import annotations

import logging
import warnings
from contextlib import contextmanager
from time import perf_counter

from alc_video.utils.logging import LOG


def configure_app_logging() -> None:
    formatter = logging.Formatter(
        "%(asctime)s %(levelname)s %(message)s", datefmt="%H:%M:%S"
    )
    for handler in LOG.handlers:
        handler.setFormatter(formatter)
    # Keep the first occurrence of these dependency deprecations visible.
    warnings.filterwarnings(
        "once",
        message=r"`torch\.backends\.cuda\.sdp_kernel\(\)` is deprecated\.",
        category=FutureWarning,
    )
    warnings.filterwarnings(
        "once",
        message="'HTTP_422_UNPROCESSABLE_ENTITY' is deprecated\\.",
        category=DeprecationWarning,
        module=r"gradio\.routes",
    )


@contextmanager
def operation(name: str, **context):
    """One start and one outcome per user operation; Gradio prints error traces."""
    started = perf_counter()
    fields = " ".join(f"{key}={value}" for key, value in context.items())
    LOG.info("%s start %s", name, fields)
    summary = {}
    try:
        yield summary
    except Exception as error:
        LOG.error(
            "%s failed elapsed=%.2fs %s error=%s: %s",
            name,
            perf_counter() - started,
            fields,
            type(error).__name__,
            error,
        )
        raise
    else:
        details = " ".join(f"{key}={value}" for key, value in summary.items())
        LOG.info(
            "%s complete elapsed=%.2fs %s %s",
            name,
            perf_counter() - started,
            fields,
            details,
        )


def timing_summary(timings: dict[str, float]) -> dict[str, str]:
    """Summarize actual measurements without repeating every representation."""
    summary = {}
    for label, prefixes in (
        ("geometry", ("geometry.",)),
        ("image_build", ("geometry_and_cache.first_frame",)),
        ("cache", ("cache.construct", "cache.reclassify")),
        ("readout", ("cache.read",)),
        ("transfer", ("cache.transfer_", "conditions.transfer_", "model.transfer_")),
        ("visualize", ("visualize.",)),
        ("generate", ("diffusion.sample",)),
        ("decode", ("video.decode",)),
        ("encode", ("preview.encode_video", "video.encode", "input.encode_video")),
        ("scene_pack", ("scene.pack",)),
    ):
        values = [value for key, value in timings.items() if key.startswith(prefixes)]
        if values:
            summary[label] = f"{sum(values):.3f}s"
    return summary
