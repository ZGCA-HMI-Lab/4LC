import os

from alc_video.utils.logging import configure_logging


def initialize_cli(log_level: str = "INFO") -> None:
    for name in (
        "DIFFUSERS_OFFLINE",
        "HF_DATASETS_OFFLINE",
        "HF_HUB_OFFLINE",
        "MODELSCOPE_OFFLINE",
        "TRANSFORMERS_OFFLINE",
    ):
        os.environ[name] = "1"
    configure_logging(log_level)
