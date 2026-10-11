"""One example.yaml per case; shared by command-line inference and the App."""

from __future__ import annotations

from dataclasses import dataclass, fields
from pathlib import Path
from typing import Any, Literal

from alc_video.conditioning.config import CacheConstructionSettings
from alc_video.inference.config import SourceInput
from alc_video.inference.controls import CameraControl, RolloutAction, TimeControl
from alc_video.runtime.profiles import DEFAULT_MODEL
from alc_video.utils.config import read_yaml, reject_unknown_fields

IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png", ".webp"}
VIDEO_SUFFIXES = {".mp4", ".webm", ".mov", ".mkv"}
TASKS = ("spatial", "temporal", "nvs", "rollout")


def media_kind(path: Path) -> Literal["image", "video"]:
    if path.suffix.lower() in IMAGE_SUFFIXES:
        return "image"
    if path.suffix.lower() in VIDEO_SUFFIXES:
        return "video"
    raise ValueError(f"Unrecognized input type: {path.name}; choose an image or video")


def _mapping(value: object, name: str) -> dict[str, Any]:
    if not isinstance(value, dict):
        raise TypeError(f"{name} must be a mapping")
    return value


def source_input(
    path: Path, kind: Literal["image", "video"], values: dict[str, Any]
) -> SourceInput:
    reject_unknown_fields(
        values,
        (f.name for f in fields(SourceInput) if f.name not in ("path", "type")),
        name="prepare",
    )
    return SourceInput(
        path=path,
        type=kind,
        start=float(values.get("start", 0)),
        sampling_fps=None
        if values.get("sampling_fps") is None
        else float(values["sampling_fps"]),
        crop=None if values.get("crop") is None else tuple(values["crop"]),
    )


@dataclass(frozen=True)
class ExampleCase:
    path: Path
    task: str
    source: SourceInput
    model: Path
    prompts: tuple[str, ...]
    camera: CameraControl
    time: TimeControl
    actions: tuple[RolloutAction, ...] = ()
    revisit: bool = False
    cache: CacheConstructionSettings | None = None

    @property
    def mode(self) -> str:
        return "control" if self.source.type == "video" else "geometry_i2v"


def load_example(path: str | Path) -> ExampleCase:
    path = Path(path).expanduser().resolve()
    if path.suffix not in (".yaml", ".yml"):
        path = path / "example.yaml"
    raw = read_yaml(path)
    task = raw.get("task")
    if task not in TASKS:
        raise ValueError(f"task={task!r}; choose one of {TASKS}")
    allowed = {"task", "input", "model", "prompt", "prepare", "cache"}
    allowed |= {"actions", "revisit"} if task == "rollout" else {"camera", "time"}
    reject_unknown_fields(raw, allowed)
    if not isinstance(raw.get("input"), str):
        raise ValueError("input must name the case's image or video file")
    input_path = (path.parent / Path(raw["input"]).expanduser()).resolve()
    kind = media_kind(input_path)
    expected = "image" if task in ("nvs", "rollout") else "video"
    if kind != expected:
        raise ValueError(
            f"task={task!r} requires {expected} input, got {input_path.name}"
        )
    source = source_input(input_path, kind, _mapping(raw.get("prepare", {}), "prepare"))
    model_name = raw.get("model", DEFAULT_MODEL.stem)
    if not isinstance(model_name, str):
        raise TypeError("model must be a profile name or a YAML path")
    model_path = Path(model_name).expanduser()
    model = (
        (path.parent / model_path)
        if model_path.suffix in (".yaml", ".yml")
        else (DEFAULT_MODEL.parent / f"{model_name}.yaml")
    ).resolve()
    prompt = raw.get("prompt", "")
    if isinstance(prompt, str):
        prompts = (prompt,)
    elif (
        task == "rollout"
        and isinstance(prompt, list)
        and prompt
        and all(isinstance(item, str) for item in prompt)
    ):
        prompts = tuple(prompt)
    else:
        raise TypeError("prompt must be text, or a per-window list of text for rollout")
    camera = CameraControl.parse(
        _mapping(raw.get("camera", {}), "camera"),
        default="fixed" if kind == "image" else "original",
    )
    camera.validate(image=kind == "image")
    time = TimeControl.parse(_mapping(raw.get("time", {}), "time"))
    time.validate(image=kind == "image")
    actions: tuple[RolloutAction, ...] = ()
    revisit = raw.get("revisit", False)
    if not isinstance(revisit, bool):
        raise TypeError("revisit must be true or false")
    if task == "rollout":
        values = raw.get("actions")
        if not isinstance(values, list) or not values:
            raise ValueError("rollout actions must be a nonempty list")
        actions = tuple(
            RolloutAction.parse(_mapping(value, f"actions[{i}]"), name=f"actions[{i}]")
            for i, value in enumerate(values)
        )
    return ExampleCase(
        path,
        task,
        source,
        model,
        prompts,
        camera,
        time,
        actions,
        revisit,
        CacheConstructionSettings.parse(raw["cache"]) if "cache" in raw else None,
    )


def discover_cases(directory: Path) -> list[ExampleCase]:
    return [load_example(path) for path in sorted(directory.rglob("example.yaml"))]
