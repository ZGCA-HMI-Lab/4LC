"""Export the learned weights of a training checkpoint for inference."""

from __future__ import annotations

import argparse
from pathlib import Path


def export_model(training_output, output):
    from alc_video.models.checkpoint import checkpoint_step, export_checkpoint_release
    from alc_video.training.execution.checkpoint import resolve_latest_checkpoint

    source = Path(training_output).expanduser().absolute()
    if source.name == "latest":
        checkpoint, _ = resolve_latest_checkpoint(source)
    elif source.name.startswith("step_"):
        checkpoint = source.resolve()
        checkpoint_step(checkpoint)
    else:
        checkpoint, _ = resolve_latest_checkpoint(source / "latest")
    target = Path(output).expanduser().resolve()
    target.parent.mkdir(parents=True, exist_ok=True)
    export_checkpoint_release(checkpoint, target)
    return target


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "training_output", help="Run directory, latest symlink or step directory"
    )
    parser.add_argument("--output", required=True, help="Output .safetensors file")
    args = parser.parse_args(argv)
    from alc_video.runtime.cli import initialize_cli

    initialize_cli()
    print(export_model(args.training_output, args.output))


if __name__ == "__main__":
    from alc_video.utils.logging import run_cli

    run_cli(main)
