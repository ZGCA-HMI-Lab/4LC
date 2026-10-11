from contextlib import ExitStack
from dataclasses import dataclass
from fractions import Fraction
from io import BytesIO
from zipfile import ZIP_STORED, ZipFile

import av
import numpy as np
from PIL import Image, ImageDraw, ImageFont

from server.media import encode_images

LABELS = {
    "input": "Input",
    "mask": "Coverage & Motion",
    "generated": "Generated",
    "rgb": "RGB",
    "depth": "Depth",
    "vjepa2": "V-JEPA",
    "dinov2": "DINOv2",
    "wm": "WorldMirror",
}


@dataclass(frozen=True)
class Comparison:
    video: bytes
    clips: dict[str, bytes]

    def archive(self):
        output = BytesIO()
        with ZipFile(output, "w", compression=ZIP_STORED) as archive:
            for name, data in self.clips.items():
                archive.writestr(f"{name}.mp4", data)
        return output.getvalue()


def compose_comparison(clips, fps, image_size, frame_count):
    """Stream synchronized frames into a labeled 2×3 or 3×3 MP4."""
    ordered = {name: clips[name] for name in LABELS if name in clips}
    if len(ordered) not in (6, 8):
        raise ValueError(
            "Comparison requires input, mask, result and 3 or 5 representations"
        )
    height, width = image_size
    label_height = 28
    rows = 2 if len(ordered) == 6 else 3
    canvas = Image.new("RGB", (3 * width, rows * (height + label_height)), (24, 33, 39))
    draw = ImageDraw.Draw(canvas)
    font = ImageFont.load_default(size=16)
    for index, name in enumerate(ordered):
        x, y = (index % 3) * width, (index // 3) * (height + label_height)
        draw.rectangle(
            (x, y, x + width - 1, y + label_height - 1), fill=(242, 244, 246)
        )
        draw.text((x + 10, y + 5), LABELS[name], font=font, fill=(37, 42, 49))
    background = np.array(canvas)
    with ExitStack() as resources:
        containers = [
            resources.enter_context(av.open(BytesIO(data))) for data in ordered.values()
        ]
        streams = [container.streams.video[0] for container in containers]
        rate = Fraction(str(fps)).limit_denominator(100000)
        if any(
            (stream.width, stream.height) != (width, height)
            or stream.average_rate != rate
            for stream in streams
        ):
            raise ValueError("Comparison clips must have matching dimensions and FPS")

        def frames():
            count = 0
            for group in zip(
                *(container.decode(video=0) for container in containers), strict=True
            ):
                result = background.copy()
                for index, frame in enumerate(group):
                    x = (index % 3) * width
                    y = (index // 3) * (height + label_height) + label_height
                    result[y : y + height, x : x + width] = frame.to_ndarray(
                        format="rgb24"
                    )
                count += 1
                yield result
            if count != frame_count:
                raise ValueError(
                    f"Comparison requires {frame_count} frames, got {count}"
                )

        data = encode_images(frames(), fps, canvas.width, canvas.height)
    return Comparison(data, ordered)
