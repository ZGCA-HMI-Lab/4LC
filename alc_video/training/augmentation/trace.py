from __future__ import annotations

from dataclasses import dataclass, replace
from typing import Any

SOURCE_DROP_ID = "source_drop"
VISIBILITY_CULLING_ID = "visibility_culling"
FIRST_FRAME_ID = "first_frame"
MOTION_THRESHOLD_ID = "motion_threshold"
PROMPT_DROP_ID = "prompt_drop"
LATENT_DROP_ID = "latent_drop"
GEOMETRY_OPERATOR_IDS = (
    SOURCE_DROP_ID,
    VISIBILITY_CULLING_ID,
)
TRANSPORT_OPERATOR_IDS = (FIRST_FRAME_ID, MOTION_THRESHOLD_ID)
CONDITION_OPERATOR_IDS = (PROMPT_DROP_ID, LATENT_DROP_ID)
EXECUTABLE_OPERATOR_IDS = (
    *GEOMETRY_OPERATOR_IDS,
    *TRANSPORT_OPERATOR_IDS,
    *CONDITION_OPERATOR_IDS,
)
AUGMENTATION_CATEGORIES = ("geometry", "transport", "condition")
OPERATOR_CATEGORY = {
    **dict.fromkeys(GEOMETRY_OPERATOR_IDS, "geometry"),
    **dict.fromkeys(TRANSPORT_OPERATOR_IDS, "transport"),
    **dict.fromkeys(CONDITION_OPERATOR_IDS, "condition"),
}
TRAIN_AUGMENTATION_METRIC_NAMES = (
    "augmentation/source_drop_applied",
    "augmentation/source_drop_ratio",
    "augmentation/source_retained_count",
    "augmentation/first_frame_applied",
    "augmentation/visibility_lateral_applied",
    "augmentation/visibility_yaw_pan_applied",
    "augmentation/visibility_pitch_pan_applied",
    "augmentation/visibility_zoom_in_applied",
    "augmentation/visibility_zoom_out_applied",
    "augmentation/visibility_anchor_retention",
    "augmentation/visibility_empty_view_fraction",
    "augmentation/motion_threshold",
    "augmentation/prompt_drop_applied",
    "augmentation/all_latents_drop_sampled",
    "augmentation/all_latents_drop_applied",
    "augmentation/latent_drop_count",
    "augmentation/condition_hit_fraction",
)

_MODIFICATION_SCOPES = {
    SOURCE_DROP_ID: ("cache.source_keyframe_valid",),
    VISIBILITY_CULLING_ID: ("cache.source_valid",),
    FIRST_FRAME_ID: ("condition.first_frame",),
    MOTION_THRESHOLD_ID: ("cache.motion_selection.threshold",),
    PROMPT_DROP_ID: ("text_context",),
    LATENT_DROP_ID: ("condition.features",),
}


@dataclass(frozen=True)
class ConditionDropout:
    prompt_keep: bool
    feature_keep: tuple[bool, ...]

    def __post_init__(self) -> None:
        if not isinstance(self.prompt_keep, bool):
            raise TypeError("prompt_keep must be bool")
        if (
            not isinstance(self.feature_keep, tuple)
            or not self.feature_keep
            or any(not isinstance(value, bool) for value in self.feature_keep)
        ):
            raise TypeError("feature_keep must be a nonempty bool tuple")
        if not self.prompt_keep and not any(self.feature_keep):
            raise ValueError("prompt or at least one readout condition must remain")

    @property
    def dropped_representation_indices(self) -> tuple[int, ...]:
        return tuple(index for index, keep in enumerate(self.feature_keep) if not keep)

    @property
    def all_features_dropped(self) -> bool:
        return not any(self.feature_keep)


@dataclass(frozen=True, slots=True)
class AugmentationDecision:
    operator_id: str
    selected: bool
    applied: bool
    parameters: dict[str, Any]
    eligible: bool
    reason: str | None = None
    metrics: dict[str, float | int] | None = None

    def __post_init__(self) -> None:
        if self.operator_id not in EXECUTABLE_OPERATOR_IDS:
            raise ValueError(f"unknown augmentation operator: {self.operator_id!r}")
        if not isinstance(self.selected, bool) or not isinstance(self.applied, bool):
            raise TypeError("augmentation selected and applied must be bool")
        if not isinstance(self.parameters, dict):
            raise TypeError("augmentation parameters must be a dict")
        if not isinstance(self.eligible, bool):
            raise TypeError("augmentation eligible must be bool")
        if self.selected and not self.eligible:
            raise ValueError("ineligible augmentation cannot be selected")
        if self.applied and not self.eligible:
            raise ValueError("ineligible augmentation cannot be applied")
        if self.reason is not None and (
            not isinstance(self.reason, str) or not self.reason
        ):
            raise ValueError("augmentation reason must be None or a non-empty string")
        if self.metrics is not None and not isinstance(self.metrics, dict):
            raise TypeError("augmentation metrics must be a dict or None")

    def as_dict(self) -> dict[str, Any]:
        return {
            "operator_id": self.operator_id,
            "selected": self.selected,
            "applied": self.applied,
            "modification_scope": list(_MODIFICATION_SCOPES[self.operator_id]),
            "parameters": dict(self.parameters),
            "eligible": self.eligible,
            "reason": self.reason,
            "metrics": None if self.metrics is None else dict(self.metrics),
        }


@dataclass(frozen=True, slots=True)
class AugmentationTrace:
    sample_id: str
    sampling_seed: int
    global_step: int
    source_count: int
    force_operator: str | None
    selected_categories: tuple[str, ...]
    decisions: tuple[AugmentationDecision, ...]

    def __post_init__(self) -> None:
        if not isinstance(self.sample_id, str) or not self.sample_id:
            raise ValueError("augmentation trace sample_id must be non-empty")
        if (
            isinstance(self.sampling_seed, bool)
            or not isinstance(self.sampling_seed, int)
            or not 0 <= self.sampling_seed < 2**64
        ):
            raise ValueError("augmentation trace sampling_seed is invalid")
        if (
            isinstance(self.global_step, bool)
            or not isinstance(self.global_step, int)
            or self.global_step < 0
        ):
            raise ValueError("augmentation trace global_step is invalid")
        if (
            isinstance(self.source_count, bool)
            or not isinstance(self.source_count, int)
            or self.source_count < 2
        ):
            raise ValueError("augmentation trace source_count must be >= 2")
        if self.force_operator is not None and self.force_operator not in (
            EXECUTABLE_OPERATOR_IDS
        ):
            raise ValueError("augmentation trace forced operator is invalid")
        if not isinstance(self.selected_categories, tuple):
            raise TypeError("augmentation trace selected_categories must be a tuple")
        if len(set(self.selected_categories)) != len(self.selected_categories):
            raise ValueError("augmentation trace selected_categories must be unique")
        if any(
            category not in AUGMENTATION_CATEGORIES
            for category in self.selected_categories
        ):
            raise ValueError("augmentation trace selected_categories are invalid")
        canonical_categories = tuple(
            category
            for category in AUGMENTATION_CATEGORIES
            if category in self.selected_categories
        )
        if self.selected_categories != canonical_categories:
            raise ValueError(
                "augmentation trace selected_categories must use canonical order"
            )
        if self.force_operator is not None and self.selected_categories != (
            OPERATOR_CATEGORY[self.force_operator],
        ):
            raise ValueError(
                "forced augmentation trace must select only its operator category"
            )
        for category, operator_ids in (("geometry", GEOMETRY_OPERATOR_IDS),):
            decisions = tuple(
                self.decision(operator_id) for operator_id in operator_ids
            )
            if sum(decision.selected for decision in decisions) > 1:
                raise ValueError(
                    f"{category} augmentation has multiple selected operators"
                )
            if sum(decision.applied for decision in decisions) > 1:
                raise ValueError(
                    f"{category} augmentation has multiple applied operators"
                )
        geometry_applied = any(
            self.decision(operator_id).applied for operator_id in GEOMETRY_OPERATOR_IDS
        )
        first_frame = self.decision(FIRST_FRAME_ID)
        if geometry_applied and first_frame.applied:
            raise ValueError("geometry and first_frame cannot both be applied")
        prompt_drop = self.decision(PROMPT_DROP_ID)
        latent_drop = self.decision(LATENT_DROP_ID)
        source_latent_keep = latent_drop.parameters.get("source_latent_keep")
        per_source_latent_keep = latent_drop.parameters.get("per_source_latent_keep")
        all_drop_sampled = latent_drop.parameters.get("all_drop_sampled")
        final_all_latents_dropped = latent_drop.parameters.get(
            "final_all_latents_dropped"
        )
        if (
            not isinstance(source_latent_keep, list)
            or not source_latent_keep
            or any(not isinstance(keep, bool) for keep in source_latent_keep)
            or not isinstance(per_source_latent_keep, list)
            or len(per_source_latent_keep) != len(source_latent_keep)
            or any(not isinstance(keep, bool) for keep in per_source_latent_keep)
            or not isinstance(all_drop_sampled, bool)
            or not isinstance(final_all_latents_dropped, bool)
            or final_all_latents_dropped != (not any(source_latent_keep))
            or (all_drop_sampled and any(source_latent_keep))
            or (not all_drop_sampled and source_latent_keep != per_source_latent_keep)
        ):
            raise ValueError("latent drop trace parameters are inconsistent")
        if prompt_drop.applied and latent_drop.applied and not any(source_latent_keep):
            raise ValueError("prompt drop and all latent drop cannot both be applied")
        decision_categories = tuple(
            category
            for category in AUGMENTATION_CATEGORIES
            if any(
                decision.selected
                and OPERATOR_CATEGORY[decision.operator_id] == category
                for decision in self.decisions
            )
        )
        if self.selected_categories != decision_categories:
            raise ValueError(
                "augmentation trace categories must match selected operators"
            )
        for decision in self.decisions:
            if decision.applied and not decision.selected:
                raise ValueError("unselected augmentation cannot be applied")

    def decision(self, operator_id: str) -> AugmentationDecision:
        for decision in self.decisions:
            if decision.operator_id == operator_id:
                return decision
        raise KeyError(f"augmentation trace has no operator {operator_id!r}")

    def record_metrics(
        self,
        operator_id: str,
        metrics: dict[str, float | int],
    ) -> AugmentationTrace:
        if not isinstance(metrics, dict):
            raise TypeError("augmentation metrics must be a dict")
        self.decision(operator_id)
        return replace(
            self,
            decisions=tuple(
                replace(decision, metrics=dict(metrics))
                if decision.operator_id == operator_id
                else decision
                for decision in self.decisions
            ),
        )

    def as_dict(self) -> dict[str, Any]:
        return {
            "sample_id": self.sample_id,
            "sampling_seed": self.sampling_seed,
            "global_step": self.global_step,
            "source_count": self.source_count,
            "force_operator": self.force_operator,
            "selected_categories": list(self.selected_categories),
            "decisions": [decision.as_dict() for decision in self.decisions],
        }


def augmentation_training_metrics(
    trace: AugmentationTrace | None,
) -> dict[str, float]:
    metrics = dict.fromkeys(TRAIN_AUGMENTATION_METRIC_NAMES, 0.0)
    if trace is None:
        return metrics

    source = trace.decision(SOURCE_DROP_ID)
    visibility = trace.decision(VISIBILITY_CULLING_ID)
    first_frame = trace.decision(FIRST_FRAME_ID)
    motion = trace.decision(MOTION_THRESHOLD_ID)
    prompt = trace.decision(PROMPT_DROP_ID)
    latent = trace.decision(LATENT_DROP_ID)

    metrics["augmentation/source_drop_applied"] = float(source.applied)
    metrics["augmentation/source_retained_count"] = float(trace.source_count)
    if source.applied:
        metrics["augmentation/source_drop_ratio"] = float(
            source.parameters["drop_ratio"]
        )
        metrics["augmentation/source_retained_count"] = float(
            source.parameters["retained_source_count"]
        )
    metrics["augmentation/first_frame_applied"] = float(first_frame.applied)
    if visibility.applied:
        strategy = str(visibility.parameters["strategy"])
        strategy_metric = {
            "lateral_look_at_v1": "augmentation/visibility_lateral_applied",
            "yaw_pan_v1": "augmentation/visibility_yaw_pan_applied",
            "pitch_pan_v1": "augmentation/visibility_pitch_pan_applied",
            "zoom_in_v1": "augmentation/visibility_zoom_in_applied",
            "zoom_out_v1": "augmentation/visibility_zoom_out_applied",
        }[strategy]
        metrics[strategy_metric] = 1.0
    if visibility.metrics is not None:
        visibility_metrics = visibility.metrics or {}
        metrics["augmentation/visibility_anchor_retention"] = float(
            visibility_metrics.get("retained_fraction", 0.0)
        )
        metrics["augmentation/visibility_empty_view_fraction"] = (
            float(visibility_metrics.get("empty_view_count", 0)) / trace.source_count
        )
    if motion.applied:
        metrics["augmentation/motion_threshold"] = float(
            motion.parameters["motion_threshold"]
        )
    metrics["augmentation/prompt_drop_applied"] = float(prompt.applied)
    source_keep = tuple(latent.parameters["source_latent_keep"])
    if not latent.applied:
        source_keep = (True,) * len(source_keep)
    dropped_count = sum(not keep for keep in source_keep)
    metrics["augmentation/all_latents_drop_sampled"] = float(
        latent.parameters["all_drop_sampled"]
    )
    metrics["augmentation/all_latents_drop_applied"] = float(
        dropped_count == len(source_keep)
    )
    metrics["augmentation/latent_drop_count"] = float(dropped_count)
    for decision in (source, visibility, first_frame):
        if decision.applied and decision.metrics is not None:
            metrics["augmentation/condition_hit_fraction"] = float(
                decision.metrics.get("condition_hit_fraction", 0.0)
            )
            break
    return metrics
