import argparse
from dataclasses import replace
from pathlib import Path
from typing import Literal, cast

from alc_video.inference.config import (
    GenerationConfig,
    InferenceRequest,
    RolloutRequest,
    SamplingConfig,
    SourceInput,
    load_generation_config,
)
from alc_video.runtime.profiles import DEFAULT_CACHE, DEFAULT_MODEL
from alc_video.utils.logging import LOG_LEVELS


def parse_frame_indices(value: str) -> tuple[int, ...]:
    parts = value.split(",")
    if any(not part.strip().isdecimal() for part in parts):
        raise argparse.ArgumentTypeError(
            "frame indices must be comma-separated nonnegative integers"
        )
    indices = tuple(int(part) for part in parts)
    if tuple(sorted(set(indices))) != indices:
        raise argparse.ArgumentTypeError(
            "frame indices must be strictly increasing and unique"
        )
    return indices


def _common_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    parser.add_argument(
        "--model", default=str(DEFAULT_MODEL), help="Shared model profile"
    )
    parser.add_argument("--input", required=True, help="Source image or video")
    parser.add_argument("--start-frame", type=int, default=0, help="First source frame")
    parser.add_argument("--num-frames", type=int, default=81, help="Source frame count")
    parser.add_argument(
        "--cache", default=str(DEFAULT_CACHE), help="Cache configuration"
    )
    parser.add_argument("--seed", type=int, default=0, help="Random seed")
    parser.add_argument("--device", default="cuda:0", help="Compute device")
    parser.add_argument("--steps", type=int, help="Sampling steps; default: 4")
    parser.add_argument(
        "--weights",
        help="Override release weights (default: local-assets/models/4lc-video-<model profile>/4lc-video.safetensors)",
    )
    group = parser.add_mutually_exclusive_group()
    group.add_argument(
        "--lightx2v",
        help="Override LightX2V weights (default: local-assets/models/lightx2v/Wan21_T2V_14B_lightx2v_cfg_step_distill_lora_rank64.safetensors)",
    )
    group.add_argument("--no-lightx2v", action="store_true", help="Disable LightX2V")
    parser.add_argument(
        "--vae-tiled",
        action=argparse.BooleanOptionalAction,
        default=None,
        help="Tiled VAE; default: false",
    )
    parser.add_argument(
        "--control-scale",
        type=float,
        help="Override model.control_scale from the selected model profile",
    )
    parser.add_argument(
        "--control-scales",
        type=float,
        nargs="+",
        help="Per-sampling-step control scales",
    )
    parser.add_argument(
        "--log-level", choices=LOG_LEVELS, default="INFO", help="Text logging level"
    )
    return parser


def parse_inference_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = _common_parser()
    parser.add_argument("--prompt", required=True, help="Text prompt")
    parser.add_argument(
        "--output", default="outputs/inference/generated.mp4", help="Output video"
    )
    parser.add_argument(
        "--source-type", choices=("image", "video"), default="video", help="Source type"
    )
    parser.add_argument(
        "--mode",
        choices=("control", "i2v", "geometry_i2v"),
        default="control",
        help="Conditioning mode",
    )
    parser.add_argument(
        "--camera", help="Target camera file; otherwise use source cameras"
    )
    parser.add_argument(
        "--num-keyframes",
        type=int,
        default=21,
        help="Uniform source keyframe count, including endpoints",
    )
    parser.add_argument(
        "--drop-source-frames",
        type=parse_frame_indices,
        default=(),
        help="Mask source-view indices after encoding",
    )
    parser.add_argument(
        "--transport",
        choices=("velocity", "identity"),
        default="velocity",
        help="Point transport",
    )
    parser.add_argument(
        "--propagate-rgb",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Propagate RGB",
    )
    return parser.parse_args(argv)


def parse_rollout_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = _common_parser()
    parser.add_argument(
        "--prompt", required=True, action="append", help="One prompt or one per chunk"
    )
    parser.add_argument(
        "--output", default="outputs/rollout/generated.mp4", help="Output video"
    )
    parser.add_argument(
        "--source-type", choices=("image", "video"), default="image", help="Source type"
    )
    parser.add_argument(
        "--mode",
        choices=("i2v", "geometry_i2v"),
        default="geometry_i2v",
        help="Conditioning mode",
    )
    parser.add_argument("--camera", required=True, help="Target camera trajectory")
    parser.add_argument(
        "--chunk-size",
        type=int,
        default=81,
        help="Frames per chunk; (T-1)/4+1 keyframes match Wan condition slots. 81 frames sample 21 and write up to 20 new keyframes",
    )
    parser.add_argument(
        "--use-cache",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Read Cache history",
    )
    parser.add_argument(
        "--order",
        choices=("earliest_first", "latest_first"),
        default="earliest_first",
        help="History traversal order",
    )
    parser.add_argument(
        "--max-entries", type=int, default=8, help="Maximum history entries"
    )
    parser.add_argument(
        "--coverage-threshold",
        type=float,
        default=0.95,
        help="History coverage stopping threshold",
    )
    parser.add_argument(
        "--resume",
        action="store_true",
        help="Skip completed output; restart incomplete output from the first frame",
    )
    return parser.parse_args(argv)


def _path(value: str) -> Path:
    return Path(value).expanduser().resolve()


def _generation(args: argparse.Namespace) -> GenerationConfig:
    g = load_generation_config(args.model, cache=args.cache)
    sampling = SamplingConfig(
        steps=g.sampling.steps if args.steps is None else args.steps,
        vae_tiled=g.sampling.vae_tiled if args.vae_tiled is None else args.vae_tiled,
        control_scales=None
        if args.control_scales is None
        else tuple(args.control_scales),
    )
    model = (
        g.model
        if args.control_scale is None
        else replace(g.model, control_scale=args.control_scale)
    )
    return replace(
        g,
        model=model,
        sampling=sampling,
        seed=args.seed,
        device=args.device,
        weights=g.weights if args.weights is None else _path(args.weights),
        lightx2v=None
        if args.no_lightx2v
        else (g.lightx2v if args.lightx2v is None else _path(args.lightx2v)),
    )


def _source(args: argparse.Namespace) -> SourceInput:
    return SourceInput(
        _path(args.input),
        cast(Literal["image", "video"], args.source_type),
        args.start_frame,
        args.num_frames,
    )


def inference_request(args: argparse.Namespace) -> InferenceRequest:
    return InferenceRequest(
        generation=_generation(args),
        source=_source(args),
        prompt=args.prompt,
        output=_path(args.output),
        camera=None if args.camera is None else _path(args.camera),
        mode=args.mode,
        num_keyframes=args.num_keyframes,
        transport=args.transport,
        drop_source_frames=args.drop_source_frames,
        propagate_rgb=args.propagate_rgb,
    )


def rollout_request(args: argparse.Namespace) -> RolloutRequest:
    return RolloutRequest(
        generation=_generation(args),
        source=_source(args),
        prompts=tuple(args.prompt),
        camera=_path(args.camera),
        output=_path(args.output),
        mode=args.mode,
        chunk_size=args.chunk_size,
        use_cache=args.use_cache,
        order=args.order,
        max_entries=args.max_entries,
        coverage_threshold=args.coverage_threshold,
    )
