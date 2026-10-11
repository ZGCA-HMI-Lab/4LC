export type ArrayRange = { offset: number; length: number };
type CloudRanges = { positions: ArrayRange; colors: ArrayRange };
export type SceneDescriptor = {
  url: string;
  static: CloudRanges;
  dynamic: CloudRanges & {
    forward?: ArrayRange;
    backward?: ArrayRange;
    sources?: ArrayRange;
  };
  source_times: number[];
  focus_distance: number;
  bounds: [number[], number[]];
};
type Cloud = { positions: Float32Array; colors: Uint8Array };
export type SceneData = {
  static: Cloud;
  dynamic: Cloud & {
    forward: Float32Array;
    backward: Float32Array;
    sources: Uint16Array;
  };
  sourceTimes: number[];
  focusDistance: number;
  radius: number;
};

export function decodeScene(
  metadata: SceneDescriptor,
  buffer: ArrayBuffer,
): SceneData {
  const floats = (range: ArrayRange) =>
    new Float32Array(buffer, range.offset, range.length);
  const cloud = (ranges: CloudRanges): Cloud => ({
    positions: floats(ranges.positions),
    colors: new Uint8Array(buffer, ranges.colors.offset, ranges.colors.length),
  });
  const moving = metadata.dynamic.positions.length > 0;
  return {
    static: cloud(metadata.static),
    dynamic: {
      ...cloud(metadata.dynamic),
      forward: moving ? floats(metadata.dynamic.forward!) : new Float32Array(),
      backward: moving
        ? floats(metadata.dynamic.backward!)
        : new Float32Array(),
      sources: moving
        ? new Uint16Array(
            buffer,
            metadata.dynamic.sources!.offset,
            metadata.dynamic.sources!.length,
          )
        : new Uint16Array(),
    },
    sourceTimes: metadata.source_times,
    focusDistance: metadata.focus_distance,
    radius:
      Math.hypot(
        ...metadata.bounds[1].map((v, i) => v - metadata.bounds[0][i]),
      ) / 2,
  };
}

// Same nearest-left / nearest-right selection as propagate_classified_anchors.
export function sourceBracket(times: number[], time: number): [number, number] {
  const target = Math.fround(time);
  let left = 0,
    right = times.length - 1;
  while (left < right) {
    const middle = Math.floor((left + right) / 2);
    if (times[middle] < target) left = middle + 1;
    else right = middle;
  }
  return times[left] === target || left === 0 ? [left, left] : [left - 1, left];
}
