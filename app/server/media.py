from fractions import Fraction
from io import BytesIO

import av
import torch


def encode_video(frames: torch.Tensor, fps: float) -> bytes:
    """Encode session results in memory; no shared or temporary output files."""
    images = frames.detach().cpu().numpy()
    return encode_images(images, fps, images.shape[2], images.shape[1])


def encode_images(images, fps, width, height) -> bytes:
    output = BytesIO()
    with av.open(output, mode="w", format="mp4") as container:
        stream = container.add_stream(
            "libx264", rate=Fraction(str(fps)).limit_denominator(100000)
        )
        stream.width, stream.height = width, height
        stream.pix_fmt = "yuv420p"
        # Short independent frames make timeline seeking responsive.
        stream.options = {"crf": "18", "preset": "fast", "g": "1"}
        for image in images:
            for packet in stream.encode(
                av.VideoFrame.from_ndarray(image, format="rgb24")
            ):
                container.mux(packet)
        for packet in stream.encode():
            container.mux(packet)
    return output.getvalue()
