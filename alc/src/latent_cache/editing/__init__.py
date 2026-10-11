"""Optional, non-destructive editing of latent-cache data."""

from latent_cache.editing.composition import compose
from latent_cache.editing.data import CacheData
from latent_cache.editing.operations import merge, select, transform
from latent_cache.editing.readout import read

__all__ = ["CacheData", "select", "transform", "merge", "read", "compose"]
