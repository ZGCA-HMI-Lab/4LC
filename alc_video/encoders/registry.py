from dataclasses import dataclass
from typing import Literal


@dataclass(frozen=True)
class EncoderBackend:
    module: str
    class_name: str
    path_kind: Literal["file", "directory"]


BACKENDS = {
    "dinov2": EncoderBackend("dinov2", "DINOv2Encoder", "directory"),
    "flux2_vae": EncoderBackend("flux2_vae", "Flux2VAEEncoder", "directory"),
    "vjepa2": EncoderBackend("vjepa2", "VJEPA2Encoder", "file"),
    "vggt": EncoderBackend("vggt", "VGGTEncoder", "directory"),
    "wan_vae": EncoderBackend("wan_vae", "WanVAESourceEncoder", "file"),
    "worldmirror": EncoderBackend("worldmirror", "WorldMirrorFeatureEncoder", "file"),
    "sd_vae": EncoderBackend("sd_vae", "SDVAEEncoder", "directory"),
    "resnet50_layer2": EncoderBackend("resnet", "ResNet50Layer2Encoder", "file"),
    "vit": EncoderBackend("vit", "ViTEncoder", "directory"),
    "depth_anything_v2": EncoderBackend(
        "depth_anything_v2", "DepthAnythingV2Encoder", "file"
    ),
}
