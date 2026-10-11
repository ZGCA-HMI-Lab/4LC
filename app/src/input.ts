export const OUTPUT_FRAMES = 81;
export const DEFAULT_FPS = 16;

export type MediaInfo = {
  media_path: string;
  kind: "image" | "video";
  width: number;
  height: number;
  duration: number;
  fps: number | null;
  frames: number;
};
export type Crop = [number, number, number, number];
export type InputSettings = {
  media_path: string;
  start: number;
  sampling_fps: number;
  crop: Crop;
};
export type CropHandle =
  | "move"
  | "nw"
  | "ne"
  | "sw"
  | "se"
  | "n"
  | "e"
  | "s"
  | "w";

export function centeredCrop(
  width: number,
  height: number,
  aspect: number,
): Crop {
  const h = Math.min(height, width / aspect);
  const w = Math.round(h * aspect);
  const roundedH = Math.round(h);
  return [
    Math.round((width - w) / 2),
    Math.round((height - roundedH) / 2),
    w,
    roundedH,
  ];
}

export function dragCrop(
  crop: Crop,
  handle: CropHandle,
  dx: number,
  dy: number,
  width: number,
  height: number,
  aspect: number,
): Crop {
  const [x, y, w, h] = crop;
  if (handle === "move")
    return [
      Math.round(Math.max(0, Math.min(width - w, x + dx))),
      Math.round(Math.max(0, Math.min(height - h, y + dy))),
      w,
      h,
    ];
  if (handle === "e" || handle === "w") {
    const centerY = y + h / 2;
    const anchorX = handle === "e" ? x : x + w;
    const limit = Math.min(
      handle === "e" ? width - anchorX : anchorX,
      2 * Math.min(centerY, height - centerY) * aspect,
    );
    const nextW = Math.round(
      Math.max(
        Math.min(aspect, limit),
        Math.min(limit, w + (handle === "e" ? dx : -dx)),
      ),
    );
    const nextH = Math.round(nextW / aspect);
    return [
      handle === "e" ? anchorX : anchorX - nextW,
      Math.round(centerY - nextH / 2),
      nextW,
      nextH,
    ];
  }
  if (handle === "n" || handle === "s") {
    const centerX = x + w / 2;
    const anchorY = handle === "s" ? y : y + h;
    const limit = Math.min(
      handle === "s" ? height - anchorY : anchorY,
      (2 * Math.min(centerX, width - centerX)) / aspect,
    );
    const nextH = Math.round(
      Math.max(
        Math.min(1, limit),
        Math.min(limit, h + (handle === "s" ? dy : -dy)),
      ),
    );
    const nextW = Math.round(nextH * aspect);
    return [
      Math.round(centerX - nextW / 2),
      handle === "s" ? anchorY : anchorY - nextH,
      nextW,
      nextH,
    ];
  }
  const sx = handle.endsWith("e") ? 1 : -1;
  const sy = handle.startsWith("s") ? 1 : -1;
  const anchorX = sx > 0 ? x : x + w;
  const anchorY = sy > 0 ? y : y + h;
  const maxWidth = Math.min(
    sx > 0 ? width - anchorX : anchorX,
    (sy > 0 ? height - anchorY : anchorY) * aspect,
  );
  const nextWidth = Math.round(
    Math.max(
      Math.min(aspect, maxWidth),
      Math.min(
        maxWidth,
        w + (sx * dx + (sy * dy) / aspect) / (1 + 1 / (aspect * aspect)),
      ),
    ),
  );
  const nextHeight = Math.round(nextWidth / aspect);
  return [
    sx > 0 ? anchorX : anchorX - nextWidth,
    sy > 0 ? anchorY : anchorY - nextHeight,
    nextWidth,
    nextHeight,
  ];
}

export function inputEnd(start: number, fps: number) {
  return start + (OUTPUT_FRAMES - 1) / fps;
}

export function defaultSamplingFps(info: MediaInfo, start = 0) {
  return info.kind === "video"
    ? (OUTPUT_FRAMES - 1) / (info.duration - start)
    : DEFAULT_FPS;
}

export function inputProblem(info: MediaInfo, start: number, fps: number) {
  if (![start, fps].every(Number.isFinite))
    return "Enter a number in each field.";
  if (fps <= 0) return "Sampling FPS must be greater than zero.";
  if (start < 0) return "Start must be at or after 0 s.";
  const end = inputEnd(start, fps);
  if (end > info.duration + 1e-5)
    return `Selected range ends at ${end.toFixed(2)} s; the video ends at ${info.duration.toFixed(2)} s. Adjust the start or sampling FPS.`;
  return null;
}
