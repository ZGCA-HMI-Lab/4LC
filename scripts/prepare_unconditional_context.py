import argparse
from pathlib import Path


def main(argv=None):
    assets = Path(__file__).resolve().parents[1] / "local-assets"
    parser = argparse.ArgumentParser(
        description="Encode the empty Wan prompt as the training unconditional context.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--wan-model-root",
        type=Path,
        default=assets / "models/Wan2.1-T2V-14B",
        help="Local Wan directory containing the text encoder and tokenizer",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=assets / "augmentation/wan21-unconditional-context.safetensors",
        help="Output context file",
    )
    parser.add_argument("--device", default="cuda:0", help="Text encoder device")
    args = parser.parse_args(argv)
    from alc_video.runtime.cli import initialize_cli
    from alc_video.runtime.resources import release_resource

    initialize_cli()
    output = args.output.expanduser().absolute()
    if output.exists() or output.is_symlink():
        raise FileExistsError(f"unconditional context already exists: {output}")

    from alc_video.models.video_model.loading import (
        WAN_TEXT_ENCODER_RELATIVE_PATH,
        WAN_TEXT_TOKENIZER_RELATIVE_PATH,
    )
    from alc_video.models.video_model.text import WanTextEncoder
    from alc_video.training.augmentation.unconditional_context import (
        write_unconditional_context,
    )

    root = args.wan_model_root.expanduser().resolve()
    encoder = WanTextEncoder.from_local_checkpoint(
        root / WAN_TEXT_ENCODER_RELATIVE_PATH,
        root / WAN_TEXT_TOKENIZER_RELATIVE_PATH,
        device=args.device,
    )
    with release_resource(encoder.offload):
        write_unconditional_context(output, encoder.encode_unconditional())
    print(output)


if __name__ == "__main__":
    from alc_video.utils.errors import input_errors
    from alc_video.utils.logging import run_cli

    def run():
        with input_errors():
            main()

    run_cli(run)
