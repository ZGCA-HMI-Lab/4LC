from __future__ import annotations

import json
from dataclasses import asdict
from pathlib import Path

from alc_video.inference.config import InferenceRequest, Request, RolloutRequest
from alc_video.inference.controls import control_values
from alc_video.utils.errors import input_errors
from alc_video.utils.logging import LOG


def _json_value(value: object) -> object:
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, dict):
        return {key: _json_value(item) for key, item in value.items()}
    if isinstance(value, (tuple, list)):
        return [_json_value(item) for item in value]
    return value


def request_config(request: Request) -> dict[str, object]:
    g = request.generation
    encoders = asdict(g.encoders)
    encoders["representations"] = dict(g.encoders.representations)
    source = asdict(request.source)
    condition: dict[str, object] = {"mode": request.mode}
    rollout: dict[str, object] = {
        "chunk_size": 81,
        "use_cache": True,
        "order": "earliest_first",
        "max_entries": 8,
        "coverage_threshold": 0.95,
    }
    if isinstance(request, InferenceRequest):
        condition.update(
            camera=control_values(request.camera), time=asdict(request.time)
        )
        source.update(
            keyframes=request.keyframes,
            num_keyframes=request.num_keyframes,
            transport=request.transport,
            drop_source_frames=request.drop_source_frames,
        )
        condition.update(prompt=request.prompt, propagate_rgb=request.propagate_rgb)
    else:
        condition["prompt"] = (
            request.prompts[0] if len(request.prompts) == 1 else request.prompts
        )
        rollout = {name: getattr(request, name) for name in rollout}
        rollout.update(
            actions=[
                {**control_values(action.camera), "chunks": action.chunks}
                for action in request.actions
            ],
            revisit=request.revisit,
        )
    value = {
        "runtime": {"seed": g.seed, "device": g.device},
        "model": {**asdict(g.model), "weights": g.weights, "lightx2v": g.lightx2v},
        "encoders": encoders,
        "cache": asdict(g.cache),
        "source": source,
        "condition": condition,
        "sampling": asdict(g.sampling),
        "output": {"path": request.output},
        "rollout": rollout,
    }
    return {key: _json_value(item) for key, item in value.items()}


def matches_saved_config(saved: dict[str, object], request: Request) -> bool:
    current = request_config(request)
    if saved != current:
        return False
    # Mapping order selects the trained representation projectors.
    saved_roles = saved["encoders"]
    assert isinstance(saved_roles, dict)
    roles = saved_roles["representations"]
    assert isinstance(roles, dict)
    return tuple(roles.items()) == request.generation.encoders.representations


def save_result_config(request: Request, frames: int) -> None:
    request.output.with_suffix(".json").write_text(
        json.dumps({"config": request_config(request), "frames": frames}, indent=2)
    )


def completed_rollout_output(request: RolloutRequest) -> bool:
    record_path = request.output.with_suffix(".json")
    if not record_path.is_file():
        return False
    try:
        record = json.loads(record_path.read_text())
    except json.JSONDecodeError as error:
        LOG.warning(
            "rollout.resume invalid record path=%s error=%s", record_path, error
        )
        return False
    with input_errors():
        if not matches_saved_config(record["config"], request):
            raise ValueError("rollout resume record differs from this run")
    return request.output.is_file()
