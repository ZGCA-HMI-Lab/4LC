# Latent Cache

**Construct once. Read at new times and viewpoints. Compose observations.** The `latent-cache` library requires Python 3.10+ and PyTorch; callers provide visual encoders, geometry and motion. Video generation and training are covered in the [project README](../README.md#4lc-video).

```bash
# From the repository root
python -m pip install -e ./alc
```

[Quick Start](#quick-start) · [Configuration](#configuration) · [Inputs & Encoders](#inputs-and-encoders) · [Editing](#editing)

## Quick Start

This example uses RGB values as features. The first observation covers half the image; a second observation supplies the missing region.

```python
import torch
from latent_cache import (
    LatentCache, LatentCacheConfig, Representation,
)

class PixelEncoder:
    representation = Representation(
        (4, 4), (1, 1), (0, 0), 3, "selected",
    )

    def encode(self, frames, *, device):
        return frames.to(device)

image_size = (4, 4)
rgb_encoder = PixelEncoder()
times = torch.tensor([[0.0, 0.5, 1.0]])
intrinsics = torch.eye(3).repeat(1, 3, 1, 1)
c2w = torch.eye(4).repeat(1, 3, 1, 1)
valid = torch.ones(1, 3, 4, 4, dtype=torch.bool)
valid[..., 2:] = False
observations = dict(
    frames={"rgb": torch.full((1, 3, 4, 4, 3), 0.5)},
    depth=torch.ones(1, 3, 4, 4), valid=valid,
    intrinsics=intrinsics, c2w=c2w, times=times,
    frame_indices=torch.tensor([[0, 1, 2]]), num_frames=3,
    fps=2.0, timeline_times=times,
)
more_observations = {
    **observations, "valid": torch.ones_like(valid),
}
query_times, query_intr = times, intrinsics
camera_path = c2w.clone()
camera_path[0, :, 0, 3] = torch.tensor([0.0, 0.125, 0.25])

config = LatentCacheConfig(
    image_size=image_size, native_resolution=True,
)
cache = LatentCache(config, {"rgb": (rgb_encoder, "rgb")})
entry = cache.construct(**observations)
views = cache.read(entry, query_times, query_intr, camera_path)
frozen_times = query_times[:, :1].expand_as(query_times)
frozen = cache.read(entry, frozen_times, query_intr, camera_path)

cache.construct(**more_observations)
composition = cache.compose(query_times, query_intr, camera_path)
print(views["rgb"].features.shape)  # [1, 3, 4, 4, 3]
print(composition.trace.coverage_rates)
```

| Operation | Result |
| :--- | :--- |
| `construct` | Appends and returns a `CacheEntry` |
| `read` | Returns one `Readout` per representation at the requested views and times |
| `compose` | Fills gaps across static history; returns `readouts` and a coverage `trace` |

`read(..., static=True)` excludes moving content. Composition keeps the first valid feature in the selected history order. Pass `entries=`, `order=`, `max_entries=` and `coverage_threshold=` to control its history and stopping condition; these are operation arguments.

## Configuration

Pass a `LatentCacheConfig` to the constructor. The video application's [cache.yaml](../configs/cache.yaml) exposes the same fields.

| Field | Default | Effect and units |
| :--- | :--- | :--- |
| `image_size` | `(336, 560)` | Source `(H,W)` in pixels; must match the encoder's spatial grid |
| `motion_threshold` | `0.1` | Projected motion in pixels per source frame; larger values classify more content as static |
| `motion_mask_dilation` | `4` | Motion-mask expansion radius in image pixels; `0` disables expansion |
| `temporal_motion_propagation` | `True` | Propagate motion-mask evidence between adjacent source keyframes |
| `temporal_motion_depth_tolerance` | `0.1` | Relative depth tolerance for that propagation; larger values allow looser depth agreement |
| `same_depth_abs_eps` | `0.001` | Absolute tolerance for near-depth candidate selection, in depth units |
| `same_depth_rel_eps` | `0.001` | Relative component of that depth tolerance |
| `target_chunk_size` | `81` | Target slots per readout batch; smaller batches reduce working memory |
| `native_resolution` | `True` | Project at the representation grid; `False` uses the image grid |
| `max_projection_window_frames` | `None` | Maximum source/target frame-index distance; `None` removes the window |

Depth selection uses `same_depth_abs_eps + same_depth_rel_eps * abs(z)`. A projection window requires source and target frame indices in a common numbering system. `target_chunk_size` batches Cache readout; the application's rollout `--chunk-size` sets video segment length.

## Inputs and Encoders

**Coordinates.** Image sizes are `(H,W)`. Use OpenCV camera axes (x right, y down, z forward), camera-to-world `c2w`, and intrinsics normalized by dividing the first row by W and the second by H. Depth is camera z-depth, velocity is in world units per second, and scene times are in seconds.

B is batch size, K geometry keyframes, M encoder-context frames, T source timeline length, and N requested frames. M equals K when no separate encoder context is supplied.

| Construction input | Shape / meaning |
| :--- | :--- |
| `frames[stream]` | Floating `[B,M,H,W,3]` in `[0,1]` |
| `depth`, `valid` | `[B,K,H,W]` depth and boolean geometry mask |
| `intrinsics`, `c2w` | `[B,K,3,3]`, `[B,K,4,4]` |
| `times`, `frame_indices` | `[B,K]` scene times and int64 source indices |
| `timeline_times`, `num_frames` | Complete `[B,T]` source times and T |
| `fps` | Positive scalar or `[B]` tensor |
| `encoding_frame_indices` | Optional `[B,M]` int64 source indices for encoder context |
| `velocity` | Optional forward/backward pair, each `[B,K,H,W,3]` |

Source indices and timeline times increase strictly; `times` equals `timeline_times` gathered at `frame_indices`. Batch items share source and encoder-context indices. Construction tensors use the depth tensor's device. Queries use times `[B,N]` and cameras `[B,N,3,3]` / `[B,N,4,4]` on the entry's device. Query times may pause or reverse while output `frame_indices` increase.

**Encoder binding.** Each representation name maps to `(encoder, stream_name)`. The encoder declares `Representation(grid, stride, offset, dim, frame_layout)` and implements `encode(frames, *, device)`. Input is `[B,F,H,W,3]` after layout expansion; output is `[B,S,h,w,D]`. Spatial grid, stride and offset use `(height,width)` order, placing cell `(i,j)` at pixel `(offset_h + i*stride_h, offset_w + j*stride_w)`.

| `frame_layout` | Encoder context | Target stride |
| :--- | :--- | :--- |
| `selected` | Selected frames independently | 1 |
| `previous_current` | Previous/current pairs; the first pair repeats the first frame | 2 |
| `causal_hold` | Full timeline, holding the latest context frame | 4 |

Each representation requires `(N-1) % stride == 0`. Use `encoding_frame_indices` for a larger shared context and `encoder_frame_indices` for per-representation subsets.

**Readout.** Features have shape `[B,J,h,w,D]`, where J is the representation's sampled target count. Coverage and provenance maps are `[B,J,h,w]`; uncovered features are zero and source indices are `-1`. Readout `valid` masks target frames, and `fill` restricts requested cells per representation.

## Editing

Continuing from Quick Start, select content, place a frozen copy, and read it with the original observation:

```python
from latent_cache import editing

placement = torch.eye(4)
placement[0, 3] = 0.25
part = editing.select(entry, valid)
placed = editing.transform(
    part, spatial=placement, time_map=lambda t: torch.zeros_like(t),
)
combined = editing.merge([entry, placed])
readouts = editing.read(
    combined, query_times, query_intr, camera_path,
    config=cache.config, fps=2.0,
)
history = editing.compose(
    [combined, cache.entries[1]],
    query_times, query_intr, camera_path,
    config=cache.config, fps=2.0, static=False,
)
```

Edits return `CacheData`, sharing features without modifying source entries. Removing geometry with `select` exposes content behind it. `feature_masks` keep the geometry but reserve its visible cells for generation, preventing history from filling them.

`transform` applies `X_new(t) = spatial(t) X_input(time_map(t))`. Supply a `[4,4]` similarity matrix or a callable returning `[4,4]` / `[B,N,4,4]`; `time_map` returns `[B,N]` source times. Callables receive query times on the query device and return tensors there.

`merge` retains all source blocks; `editing.read` resolves visibility by depth. `editing.compose` reads each history item jointly and fills gaps in order; `static=False` includes dynamic content.
