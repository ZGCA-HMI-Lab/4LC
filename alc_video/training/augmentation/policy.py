from __future__ import annotations

import math
from typing import Any

import torch

from alc_video.training.augmentation.config import AugmentationConfig
from alc_video.training.augmentation.trace import (
    AUGMENTATION_CATEGORIES,
    EXECUTABLE_OPERATOR_IDS,
    FIRST_FRAME_ID,
    GEOMETRY_OPERATOR_IDS,
    LATENT_DROP_ID,
    MOTION_THRESHOLD_ID,
    OPERATOR_CATEGORY,
    PROMPT_DROP_ID,
    SOURCE_DROP_ID,
    VISIBILITY_CULLING_ID,
    AugmentationDecision,
    AugmentationTrace,
    ConditionDropout,
)

_STREAM_IDS = {
    "geometry_gate": 100,
    "geometry_operator": 102,
    "source_drop_count": 103,
    "source_drop_ids": 104,
    "visibility_strategy": 107,
    "visibility_direction": 108,
    "visibility_severity": 109,
    "first_frame_gate": 110,
    "motion_threshold": 301,
    "prompt_drop_gate": 201,
    "all_latents_drop_gate": 202,
    "per_source_latent_drop": 203,
}
_MASK64 = (1 << 64) - 1


def _mix_seed(sampling_seed: int, global_step: int, stream_id: int) -> int:
    value = (
        int(sampling_seed)
        + (int(global_step) + 1) * 0x9E3779B97F4A7C15
        + int(stream_id) * 0xBF58476D1CE4E5B9
    ) & _MASK64
    value = ((value ^ (value >> 30)) * 0xBF58476D1CE4E5B9) & _MASK64
    value = ((value ^ (value >> 27)) * 0x94D049BB133111EB) & _MASK64
    value ^= value >> 31
    return value % (2**63)


def _generator(sampling_seed: int, global_step: int, stream: str) -> torch.Generator:
    generator = torch.Generator(device="cpu")
    generator.manual_seed(_mix_seed(sampling_seed, global_step, _STREAM_IDS[stream]))
    return generator


def _uniform(sampling_seed: int, global_step: int, stream: str) -> float:
    return float(
        torch.rand(
            (),
            generator=_generator(sampling_seed, global_step, stream),
            dtype=torch.float64,
        ).item()
    )


def _select_group(
    *,
    gate: float,
    probability: float,
    choice: float,
    branches: tuple[tuple[str, bool], ...],
) -> str | None:
    enabled = tuple(operator_id for operator_id, active in branches if active)
    if not enabled:
        return None
    if gate >= probability:
        return None
    return enabled[min(int(choice * len(enabled)), len(enabled) - 1)]


class AugmentationExecutor:
    def __init__(self, config: AugmentationConfig) -> None:
        if not isinstance(config, AugmentationConfig):
            raise TypeError("config must be AugmentationConfig")
        self.config = config

    @property
    def enabled_operator_ids(self) -> tuple[str, ...]:
        geometry = self.config.geometry
        transport = self.config.transport
        condition = self.config.condition
        enabled = {
            SOURCE_DROP_ID: (
                geometry.probability > 0.0 and geometry.source_drop.enabled
            ),
            VISIBILITY_CULLING_ID: (
                geometry.probability > 0.0 and geometry.visibility_culling.enabled
            ),
            FIRST_FRAME_ID: (
                transport.first_frame.enabled
                and transport.first_frame.probability > 0.0
            ),
            MOTION_THRESHOLD_ID: (transport.motion_threshold.enabled),
            PROMPT_DROP_ID: (condition.prompt_drop.probability > 0.0),
            LATENT_DROP_ID: (
                condition.latent_drop.all_probability > 0.0
                or condition.latent_drop.per_source_probability > 0.0
            ),
        }
        return tuple(
            operator_id
            for operator_id in EXECUTABLE_OPERATOR_IDS
            if enabled[operator_id]
        )

    def _source_drop_parameters(
        self,
        *,
        sampling_seed: int,
        global_step: int,
        source_count: int,
    ) -> dict[str, Any]:
        config = self.config.geometry.source_drop
        if not config.enabled:
            return {}
        if config.max_drop_ratio is None:
            raise RuntimeError("enabled source_drop has no max_drop_ratio")
        maximum = math.floor(source_count * config.max_drop_ratio)
        if not 1 <= maximum < source_count:
            raise ValueError(
                "source_drop max_drop_ratio produces invalid M: "
                f"K={source_count}, M={maximum}"
            )
        count = int(
            torch.randint(
                1,
                maximum + 1,
                (),
                generator=_generator(
                    sampling_seed,
                    global_step,
                    "source_drop_count",
                ),
            ).item()
        )
        retained_count = source_count - count
        dropped = torch.randperm(
            source_count,
            generator=_generator(
                sampling_seed,
                global_step,
                "source_drop_ids",
            ),
        )[:count]
        dropped_ids = sorted(int(value) for value in dropped.tolist())
        return {
            "K": source_count,
            "M": maximum,
            "m": count,
            "retained_source_count": retained_count,
            "drop_ratio": count / source_count,
            "dropped_source_ids": dropped_ids,
        }

    def _visibility_parameters(
        self,
        *,
        sampling_seed: int,
        global_step: int,
    ) -> dict[str, Any]:
        config = self.config.geometry.visibility_culling
        if not config.enabled:
            return {}
        strategies = tuple(
            (strategy, probability)
            for strategy, probability in zip(
                config.strategies,
                config.strategy_probabilities,
                strict=True,
            )
            if probability > 0.0
        )
        if not strategies:
            raise RuntimeError("enabled visibility_culling has no trajectory range")
        choice = _uniform(sampling_seed, global_step, "visibility_strategy")
        cumulative = 0.0
        strategy = strategies[-1][0]
        for candidate, probability in strategies:
            cumulative += probability
            if choice < cumulative:
                strategy = candidate
                break
        unit = _uniform(sampling_seed, global_step, "visibility_severity")

        def sample(minimum: float | None, maximum: float | None) -> float:
            if minimum is None or maximum is None:
                raise RuntimeError(f"visibility strategy {strategy!r} has no range")
            return minimum + unit * (maximum - minimum)

        parameters: dict[str, Any] = {"strategy": strategy}
        if strategy in {"lateral_look_at_v1", "yaw_pan_v1"}:
            parameters["direction"] = (
                "left"
                if _uniform(sampling_seed, global_step, "visibility_direction") < 0.5
                else "right"
            )
        elif strategy == "pitch_pan_v1":
            parameters["direction"] = (
                "up"
                if _uniform(sampling_seed, global_step, "visibility_direction") < 0.5
                else "down"
            )
        if strategy == "lateral_look_at_v1":
            parameters["translation"] = sample(
                config.lateral_translation_min,
                config.lateral_translation_max,
            )
        elif strategy == "yaw_pan_v1":
            parameters["degrees"] = sample(
                config.pan_degrees_min,
                config.pan_degrees_max,
            )
        elif strategy == "pitch_pan_v1":
            parameters["degrees"] = sample(
                config.pitch_degrees_min,
                config.pitch_degrees_max,
            )
        elif strategy == "zoom_in_v1":
            parameters["probe_focal_scale"] = sample(
                config.zoom_in_focal_scale_min,
                config.zoom_in_focal_scale_max,
            )
        elif strategy == "zoom_out_v1":
            parameters["probe_focal_scale"] = sample(
                config.zoom_out_focal_scale_min,
                config.zoom_out_focal_scale_max,
            )
        else:
            raise AssertionError(f"unknown visibility strategy {strategy!r}")
        return parameters

    def _motion_threshold_parameters(
        self,
        *,
        sampling_seed: int,
        global_step: int,
    ) -> dict[str, Any]:
        config = self.config.transport.motion_threshold
        if not config.enabled:
            return {}
        if config.minimum is None or config.maximum is None:
            raise RuntimeError("enabled motion_threshold has no range")
        unit = _uniform(sampling_seed, global_step, "motion_threshold")
        threshold = config.minimum + unit * (config.maximum - config.minimum)
        return {"motion_threshold": threshold}

    def _condition_state(
        self,
        *,
        sampling_seed: int,
        global_step: int,
        force_operator: str | None,
        source_latent_count: int,
    ) -> tuple[ConditionDropout, bool, tuple[bool, ...]]:
        if source_latent_count <= 0:
            raise ValueError("source_latent_count must be positive")
        all_kept = (True,) * source_latent_count
        all_dropped = (False,) * source_latent_count
        if force_operator == PROMPT_DROP_ID:
            return (
                ConditionDropout(
                    prompt_keep=False,
                    feature_keep=all_kept,
                ),
                False,
                all_kept,
            )
        if force_operator == LATENT_DROP_ID:
            return (
                ConditionDropout(
                    prompt_keep=True,
                    feature_keep=all_dropped,
                ),
                True,
                all_kept,
            )
        if force_operator is not None:
            return (
                ConditionDropout(
                    prompt_keep=True,
                    feature_keep=all_kept,
                ),
                False,
                all_kept,
            )
        prompt_drop = self.config.condition.prompt_drop.probability
        latent_drop = self.config.condition.latent_drop
        prompt_keep = not (
            _uniform(sampling_seed, global_step, "prompt_drop_gate") < prompt_drop
        )
        all_drop_sampled = (
            _uniform(sampling_seed, global_step, "all_latents_drop_gate")
            < latent_drop.all_probability
        )
        per_source_latent_keep = tuple(
            bool(value >= latent_drop.per_source_probability)
            for value in torch.rand(
                source_latent_count,
                generator=_generator(
                    sampling_seed,
                    global_step,
                    "per_source_latent_drop",
                ),
                dtype=torch.float64,
            ).tolist()
        )
        source_latent_keep = all_dropped if all_drop_sampled else per_source_latent_keep
        if not any(source_latent_keep):
            prompt_keep = True
        return (
            ConditionDropout(
                prompt_keep=prompt_keep,
                feature_keep=source_latent_keep,
            ),
            all_drop_sampled,
            per_source_latent_keep,
        )

    def decide(
        self,
        *,
        sample_id: str,
        sampling_seed: int,
        global_step: int,
        source_count: int,
        source_latent_count: int,
        force_operator: str | None = None,
    ) -> AugmentationTrace:
        if not isinstance(sample_id, str) or not sample_id:
            raise ValueError("augmentation sample_id must be non-empty")
        if (
            isinstance(sampling_seed, bool)
            or not isinstance(sampling_seed, int)
            or not 0 <= sampling_seed < 2**64
        ):
            raise ValueError("sampling_seed must be an unsigned 64-bit integer")
        if (
            isinstance(global_step, bool)
            or not isinstance(global_step, int)
            or global_step < 0
        ):
            raise ValueError("global_step must be a nonnegative integer")
        if (
            isinstance(source_count, bool)
            or not isinstance(source_count, int)
            or source_count < 2
        ):
            raise ValueError("source_count must be an integer >= 2")
        if (
            isinstance(source_latent_count, bool)
            or not isinstance(source_latent_count, int)
            or source_latent_count <= 0
        ):
            raise ValueError("source_latent_count must be a positive integer")
        if (
            force_operator is not None
            and force_operator not in self.enabled_operator_ids
        ):
            raise ValueError(
                f"forced operator {force_operator!r} is not enabled by the profile"
            )

        geometry = self.config.geometry
        transport = self.config.transport
        first_frame_probability = (
            transport.first_frame.probability if transport.first_frame.enabled else 0.0
        )
        selected_first_frame = (
            force_operator == FIRST_FRAME_ID
            if force_operator is not None
            else (
                first_frame_probability > 0.0
                and _uniform(sampling_seed, global_step, "first_frame_gate")
                < first_frame_probability
            )
        )
        selected_geometry = (
            _select_group(
                gate=_uniform(
                    sampling_seed,
                    global_step,
                    "geometry_gate",
                ),
                probability=geometry.probability,
                choice=_uniform(
                    sampling_seed,
                    global_step,
                    "geometry_operator",
                ),
                branches=(
                    (SOURCE_DROP_ID, geometry.source_drop.enabled),
                    (VISIBILITY_CULLING_ID, geometry.visibility_culling.enabled),
                ),
            )
            if force_operator is None and not selected_first_frame
            else force_operator
            if force_operator in GEOMETRY_OPERATOR_IDS
            else None
        )
        (
            condition_state,
            all_drop_sampled,
            per_source_latent_keep,
        ) = self._condition_state(
            sampling_seed=sampling_seed,
            global_step=global_step,
            force_operator=force_operator,
            source_latent_count=source_latent_count,
        )
        selected_prompt_drop = not condition_state.prompt_keep
        selected_latent_drop = bool(condition_state.dropped_representation_indices)

        selected_motion_threshold = (
            force_operator == MOTION_THRESHOLD_ID
            if force_operator is not None
            else transport.motion_threshold.enabled
        )
        source_parameters = (
            self._source_drop_parameters(
                sampling_seed=sampling_seed,
                global_step=global_step,
                source_count=source_count,
            )
            if selected_geometry == SOURCE_DROP_ID
            else {}
        )
        visibility_parameters = (
            self._visibility_parameters(
                sampling_seed=sampling_seed,
                global_step=global_step,
            )
            if selected_geometry == VISIBILITY_CULLING_ID
            else {}
        )
        motion_threshold_parameters = (
            self._motion_threshold_parameters(
                sampling_seed=sampling_seed,
                global_step=global_step,
            )
            if selected_motion_threshold
            else {}
        )

        def decision(
            operator_id: str,
            *,
            eligible: bool,
            selected: bool,
            parameters: dict[str, Any],
        ) -> AugmentationDecision:
            return AugmentationDecision(
                operator_id=operator_id,
                selected=selected,
                applied=selected,
                parameters=parameters,
                eligible=eligible,
                reason=None if selected else "not_selected" if eligible else "disabled",
            )

        decisions = (
            decision(
                SOURCE_DROP_ID,
                eligible=(geometry.probability > 0.0 and geometry.source_drop.enabled),
                selected=selected_geometry == SOURCE_DROP_ID,
                parameters=source_parameters,
            ),
            decision(
                VISIBILITY_CULLING_ID,
                eligible=(
                    geometry.probability > 0.0 and geometry.visibility_culling.enabled
                ),
                selected=selected_geometry == VISIBILITY_CULLING_ID,
                parameters=visibility_parameters,
            ),
            decision(
                FIRST_FRAME_ID,
                eligible=first_frame_probability > 0.0,
                selected=selected_first_frame,
                parameters=(
                    {
                        "source_index": 0,
                        "source_count": 1,
                        "use_geometry": transport.first_frame.use_geometry,
                        "propagation": (
                            "cache_camera_projection"
                            if transport.first_frame.use_geometry
                            else "none"
                        ),
                        "motion": "disabled",
                    }
                    if selected_first_frame
                    else {}
                ),
            ),
            decision(
                MOTION_THRESHOLD_ID,
                eligible=transport.motion_threshold.enabled,
                selected=selected_motion_threshold,
                parameters=motion_threshold_parameters,
            ),
            decision(
                PROMPT_DROP_ID,
                eligible=self.config.condition.prompt_drop.probability > 0.0,
                selected=selected_prompt_drop,
                parameters={
                    "probability": self.config.condition.prompt_drop.probability,
                },
            ),
            decision(
                LATENT_DROP_ID,
                eligible=(
                    self.config.condition.latent_drop.all_probability > 0.0
                    or self.config.condition.latent_drop.per_source_probability > 0.0
                ),
                selected=selected_latent_drop,
                parameters={
                    "source_latent_keep": list(condition_state.feature_keep),
                    "per_source_latent_keep": list(per_source_latent_keep),
                    "all_drop_sampled": all_drop_sampled,
                    "final_all_latents_dropped": (condition_state.all_features_dropped),
                    "all_probability": (
                        self.config.condition.latent_drop.all_probability
                    ),
                    "per_source_probability": (
                        self.config.condition.latent_drop.per_source_probability
                    ),
                },
            ),
        )
        selected_categories = tuple(
            category
            for category in AUGMENTATION_CATEGORIES
            if any(
                decision.selected
                and OPERATOR_CATEGORY[decision.operator_id] == category
                for decision in decisions
            )
        )
        return AugmentationTrace(
            sample_id=sample_id,
            sampling_seed=sampling_seed,
            global_step=global_step,
            source_count=source_count,
            force_operator=force_operator,
            selected_categories=selected_categories,
            decisions=decisions,
        )

    def condition_augmentation(
        self,
        trace: AugmentationTrace,
    ) -> ConditionDropout:
        if not isinstance(trace, AugmentationTrace):
            raise TypeError("trace must be AugmentationTrace")
        prompt = trace.decision(PROMPT_DROP_ID)
        latent = trace.decision(LATENT_DROP_ID)
        source_latent_keep = tuple(latent.parameters["source_latent_keep"])
        if not latent.applied:
            source_latent_keep = (True,) * len(source_latent_keep)
        return ConditionDropout(
            prompt_keep=not prompt.applied,
            feature_keep=source_latent_keep,
        )

    def apply_text_context(
        self,
        context: torch.Tensor,
        trace: AugmentationTrace,
        *,
        unconditional_context: torch.Tensor | None,
    ) -> torch.Tensor:
        if not isinstance(trace, AugmentationTrace):
            raise TypeError("trace must be AugmentationTrace")
        if not trace.decision(PROMPT_DROP_ID).applied:
            return context
        if not isinstance(unconditional_context, torch.Tensor):
            raise RuntimeError("prompt drop requires unconditional context")
        if context.ndim != 3 or unconditional_context.ndim not in (2, 3):
            raise ValueError("text contexts must be [B,S,C] or unconditional [S,C]")
        value = (
            unconditional_context.unsqueeze(0)
            if unconditional_context.ndim == 2
            else unconditional_context
        )
        if int(value.shape[0]) == 1 and int(context.shape[0]) != 1:
            value = value.expand(int(context.shape[0]), -1, -1)
        if tuple(value.shape) != tuple(context.shape):
            raise ValueError(
                f"unconditional context shape {tuple(value.shape)} != "
                f"{tuple(context.shape)}"
            )
        return value.to(device=context.device, dtype=context.dtype)
