from latent_cache.geometry.candidates import (
    DEPTH_EPS,
    GeometryCandidates,
    build_candidates,
)
from latent_cache.geometry.projection import (
    ProjectionResult,
    TransportPlan,
    apply_plan,
    project,
    rasterize_plan,
)
from latent_cache.geometry.spatial import SpatialGridSpec, unproject_depth

__all__ = [
    "DEPTH_EPS",
    "GeometryCandidates",
    "build_candidates",
    "SpatialGridSpec",
    "unproject_depth",
    "ProjectionResult",
    "TransportPlan",
    "apply_plan",
    "project",
    "rasterize_plan",
]
