import torch
import torch.nn.functional as F

from alc_video.models.adapter.alignment import align_wan_coverage, readout_stride
from alc_video.models.video_model.temporal import wan_vae_latent_frame_count

# These colors are shared with the workbench legend.
UNCOVERED = (24, 33, 39)
STATIC = (125, 151, 168)
DYNAMIC = (222, 147, 62)


def render_mask(readouts, image_size, frame_count):
    """Show aligned coverage and motion; motion remains diagnostic information."""
    height, width = image_size
    target = (1, 16, wan_vae_latent_frame_count(frame_count), height // 8, width // 8)
    coverage, moving = [], []
    for readout in readouts.values():
        options = {
            "grid": tuple(readout.coverage.shape[2:]),
            "readout_stride": readout_stride(readout.target_indices, frame_count),
        }
        coverage.append(align_wan_coverage(readout.coverage.cpu(), target, **options))
        moving.append(align_wan_coverage(readout.moving.cpu(), target, **options))
    covered = torch.stack(coverage).amax(0)
    dynamic = torch.stack(moving).amax(0)
    if bool((dynamic > covered).any()):
        raise ValueError("Moving coverage must be contained in hit coverage")
    weights = torch.cat((1 - covered, covered - dynamic, dynamic), dim=1)[0]
    colors = torch.tensor((UNCOVERED, STATIC, DYNAMIC), dtype=torch.float32)
    rgb = torch.einsum("cfhw,cd->fdhw", weights, colors)
    # F0 is independent; each subsequent Wan latent covers four video frames.
    slots = (torch.arange(frame_count) + 3) // 4
    rgb = F.interpolate(rgb.index_select(0, slots), size=image_size, mode="nearest")
    return rgb.permute(0, 2, 3, 1).round().clamp(0, 255).byte()
