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
from alc_video.inference.controls import (
    CAMERA_PRESETS,
    TIME_MODES,
    CameraControl,
    RolloutAction,
    TimeControl,
    control_values,
)
from alc_video.inference.examples import ExampleCase, load_example, media_kind
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
        "--model",
        help=f"Model YAML; overrides the example profile, default: {DEFAULT_MODEL}",
    )
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument("--input", help="Source image or video")
    source.add_argument("--example", help="Case directory containing example.yaml")
    parser.add_argument("--start", type=float, help="Clip start in seconds; default: 0")
    parser.add_argument(
        "--sampling-fps",
        type=float,
        help="Sample 81 frames at this rate; default: cover the remaining video, images use 16 fps",
    )
    parser.add_argument(
        "--crop",
        type=int,
        nargs=4,
        metavar=("X", "Y", "W", "H"),
        help="Source pixel crop; default: centered crop matching model aspect ratio",
    )
    parser.add_argument(
        "--cache", default=str(DEFAULT_CACHE), help="Cache configuration"
    )
    parser.add_argument("--seed", type=int, default=0, help="Random seed")
    parser.add_argument("--device", default="cuda:0", help="Compute device")
    parser.add_argument("--steps", type=int, help="Sampling steps; default: 4")
    parser.add_argument(
        "--weights",
        help="Override release weights (default: local-assets/models/4lc-video-<model profile>.safetensors)",
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


def _camera_arguments(parser: argparse.ArgumentParser) -> None:
    parser.add_argument(
        "--camera",
        dest="camera_preset",
        choices=CAMERA_PRESETS,
        help="Camera preset; inference defaults to original for video or fixed for image, rollout defaults to dolly",
    )
    parser.add_argument(
        "--direction",
        choices=("left", "right", "in", "out", "up", "down"),
        help="Camera move direction; default: right, in or up for the selected preset",
    )
    parser.add_argument(
        "--amount", type=float, help="Camera move strength from 0 to 1; default: 0.4"
    )


def parse_inference_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = _common_parser()
    parser.add_argument(
        "--prompt", help="Optional text prompt; overrides the example Prompt"
    )
    parser.add_argument(
        "--output", default="outputs/inference/generated.mp4", help="Output video"
    )
    parser.add_argument(
        "--source-type",
        choices=("image", "video"),
        help="Source type; inferred from the input file by default",
    )
    parser.add_argument(
        "--mode",
        choices=("control", "i2v", "geometry_i2v"),
        help="Conditioning mode; default: control for video, geometry_i2v for image",
    )
    _camera_arguments(parser)
    parser.add_argument(
        "--time",
        dest="time_mode",
        choices=TIME_MODES,
        help="Source video time mode; default: normal",
    )
    parser.add_argument(
        "--freeze-at",
        type=float,
        help="Freeze at seconds within the prepared clip; default: 0",
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
        "--prompt",
        action="append",
        help="Optional prompt; one for all windows or one per 81-frame window",
    )
    parser.add_argument(
        "--output", default="outputs/rollout/generated.mp4", help="Output video"
    )
    parser.add_argument(
        "--source-type",
        choices=("image",),
        default="image",
        help="Rollout starts from a single image",
    )
    parser.add_argument(
        "--mode",
        choices=("i2v", "geometry_i2v"),
        default="geometry_i2v",
        help="Conditioning mode",
    )
    _camera_arguments(parser)
    parser.add_argument(
        "--chunks",
        type=int,
        help="81-frame windows for a single camera action; default: 1. Use example.yaml actions for a multi-action route",
    )
    parser.add_argument(
        "--revisit",
        action=argparse.BooleanOptionalAction,
        default=None,
        help="Append the exact reversed camera path to revisit earlier views; doubles the window count",
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


def _generation(
    args: argparse.Namespace, example: ExampleCase | None
) -> GenerationConfig:
    model_path = args.model or (example.model if example else DEFAULT_MODEL)
    g = load_generation_config(model_path, cache=args.cache)
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


def _source(args: argparse.Namespace, example: ExampleCase | None) -> SourceInput:
    if example:
        source = example.source
        if args.source_type is not None and args.source_type != source.type:
            raise ValueError("--source-type differs from the example task's input type")
    else:
        path = _path(args.input)
        source = SourceInput(
            path, cast(Literal["image", "video"], args.source_type or media_kind(path))
        )
    updates = {
        name: getattr(args, name)
        for name in ("start", "sampling_fps", "crop")
        if getattr(args, name) is not None
    }
    if "crop" in updates:
        updates["crop"] = tuple(updates["crop"])
    return replace(source, **updates)


def _camera(
    args: argparse.Namespace, default: str, base: CameraControl | None = None
) -> CameraControl:
    values = control_values(base) if base else {}
    if args.camera_preset is not None:
        values["preset"] = args.camera_preset
        values.pop("direction", None)
    for name in ("direction", "amount"):
        if getattr(args, name) is not None:
            values[name] = getattr(args, name)
    return CameraControl.parse(values, default=default)


def inference_request(args: argparse.Namespace) -> InferenceRequest:
    example = load_example(args.example) if args.example else None
    if example and example.task == "rollout":
        raise ValueError("This example is a rollout; use scripts/rollout.py")
    source = _source(args, example)
    camera = _camera(
        args,
        "fixed" if source.type == "image" else "original",
        example.camera if example else None,
    )
    camera.validate(image=source.type == "image")
    base_time = example.time if example else TimeControl()
    time = TimeControl(
        args.time_mode if args.time_mode is not None else base_time.mode,
        args.freeze_at if args.freeze_at is not None else base_time.at,
    )
    time.validate(image=source.type == "image")
    return InferenceRequest(
        generation=_generation(args, example),
        source=source,
        prompt=args.prompt
        if args.prompt is not None
        else (example.prompts[0] if example else ""),
        output=_path(args.output),
        camera=camera,
        time=time,
        mode=args.mode or ("geometry_i2v" if source.type == "image" else "control"),
        num_keyframes=args.num_keyframes,
        transport=args.transport,
        drop_source_frames=args.drop_source_frames,
        propagate_rgb=args.propagate_rgb,
    )


def rollout_request(args: argparse.Namespace) -> RolloutRequest:
    example = load_example(args.example) if args.example else None
    if example and example.task != "rollout":
        raise ValueError("This example is a single shot; use scripts/infer.py")
    source = _source(args, example)
    if source.type != "image" or media_kind(source.path) != "image":
        raise ValueError("Rollout requires a single image input")
    if example:
        if any(
            getattr(args, key) is not None
            for key in ("camera_preset", "direction", "amount", "chunks")
        ):
            raise ValueError(
                "Edit the example's actions to change a multi-action route; single-action flags apply with --input"
            )
        actions = example.actions
    else:
        camera = _camera(args, "dolly")
        action = RolloutAction.parse(
            {
                **control_values(camera),
                "chunks": 1 if args.chunks is None else args.chunks,
            },
            name="action",
        )
        actions = (action,)
    return RolloutRequest(
        generation=_generation(args, example),
        source=source,
        prompts=tuple(args.prompt)
        if args.prompt is not None
        else (example.prompts if example else ("",)),
        actions=actions,
        revisit=args.revisit
        if args.revisit is not None
        else (example.revisit if example else False),
        output=_path(args.output),
        mode=args.mode,
        use_cache=args.use_cache,
        order=args.order,
        max_entries=args.max_entries,
        coverage_threshold=args.coverage_threshold,
    )
