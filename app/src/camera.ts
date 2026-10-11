import { Euler, Matrix4, Quaternion, Vector3 } from "three";

export type Vec3 = [number, number, number];
export type FrameSize = { width: number; height: number };
export type Camera = {
  position: Vec3;
  rotation: [number, number, number, number];
  fov: number; // Vertical field of view, in radians.
  focalRatio: number; // fx / fy in pixels.
  principal: [number, number]; // Normalized principal point.
  skewRatio: number; // Pixel skew / fy.
  focusDistance: number; // Orbit-navigation pivot distance; not a lens parameter.
};
export type CameraPreset =
  | "original"
  | "offset"
  | "fixed"
  | "dolly"
  | "zoom"
  | "truck"
  | "pedestal"
  | "pan"
  | "tilt"
  | "orbit";
export type Motion = {
  preset: CameraPreset;
  direction: -1 | 1;
  amount: number;
};
export const CAMERA_LABELS: Record<CameraPreset, string> = {
  original: "Original path",
  offset: "Offset path",
  fixed: "Fixed",
  dolly: "Dolly",
  zoom: "Zoom",
  truck: "Truck",
  pedestal: "Pedestal",
  pan: "Pan",
  tilt: "Tilt",
  orbit: "Orbit",
};
export const CUSTOM_PRESETS: CameraPreset[] = [
  "fixed",
  "dolly",
  "zoom",
  "truck",
  "pedestal",
  "pan",
  "tilt",
  "orbit",
];
// Initial lens before a source camera is loaded.
const DEFAULT_FOV = 0.6178614878474833;
export const clamp = (n: number, min: number, max: number) =>
  Math.max(min, Math.min(max, n));
export const dot = (a: Vec3, b: Vec3) =>
  a.reduce((sum, v, i) => sum + v * b[i], 0);
export const add = (a: Vec3, b: Vec3, scale = 1): Vec3 =>
  a.map((v, i) => v + b[i] * scale) as Vec3;

export function cameraPose(camera: Camera) {
  const q = new Quaternion(...camera.rotation);
  const rotate = (v: Vec3) =>
    new Vector3(...v).applyQuaternion(q).toArray() as Vec3;
  return {
    position: camera.position,
    forward: rotate([0, 0, -1]),
    right: rotate([1, 0, 0]),
    up: rotate([0, 1, 0]),
  };
}

function cameraAround(
  pivot: Vec3,
  yaw: number,
  pitch: number,
  distance: number,
  fov = DEFAULT_FOV,
): Camera {
  const camera: Camera = {
    position: pivot,
    rotation: new Quaternion()
      .setFromEuler(new Euler(-pitch, yaw, 0, "YXZ"))
      .toArray(),
    fov,
    focalRatio: 1,
    principal: [0.5, 0.5],
    skewRatio: 0,
    focusDistance: distance,
  };
  return {
    ...camera,
    position: add(pivot, cameraPose(camera).forward, -distance),
  };
}

export const INITIAL_CAMERA = cameraAround([0, 1.55, 0], -0.3, 0.23, 9);
export const INITIAL_MOTION: Motion = {
  preset: "original",
  direction: 1,
  amount: 0.4,
};
export const usesInputPath = (preset: CameraPreset) =>
  preset === "original" || preset === "offset";

export function orbitCamera(
  camera: Camera,
  yawDelta: number,
  pitchDelta: number,
): Camera {
  const pivot = add(
    camera.position,
    cameraPose(camera).forward,
    camera.focusDistance,
  );
  const next = rotateCamera(camera, yawDelta, -pitchDelta);
  return {
    ...next,
    position: add(pivot, cameraPose(next).forward, -camera.focusDistance),
  };
}

function rotateCamera(camera: Camera, yaw: number, pitch: number): Camera {
  const q = new Quaternion(...camera.rotation)
    .multiply(new Quaternion().setFromAxisAngle(new Vector3(0, 1, 0), yaw))
    .multiply(new Quaternion().setFromAxisAngle(new Vector3(1, 0, 0), pitch));
  return { ...camera, rotation: q.normalize().toArray() };
}

export function cameraIntrinsics(camera: Camera, frame: FrameSize) {
  const fy = 1 / (2 * Math.tan(camera.fov / 2));
  const xScale = (fy * frame.height) / frame.width;
  return {
    fx: xScale * camera.focalRatio,
    fy,
    cx: camera.principal[0],
    cy: camera.principal[1],
    skew: xScale * camera.skewRatio,
  };
}

// Core cameras use OpenCV axes; the browser world uses x-right, y-up, z-back.
export function cameraFromCV(
  intr: number[][],
  c2w: number[][],
  frame: FrameSize,
  focusDistance: number,
): Camera {
  const sign = [1, -1, -1, 1];
  const matrix = new Matrix4().set(
    ...(c2w.flatMap((row, i) =>
      row.map((v, j) => v * sign[i] * sign[j]),
    ) as Parameters<Matrix4["set"]>),
  );
  return {
    position: [c2w[0][3], -c2w[1][3], -c2w[2][3]],
    rotation: new Quaternion()
      .setFromRotationMatrix(matrix)
      .normalize()
      .toArray(),
    fov: 2 * Math.atan(1 / (2 * intr[1][1])),
    focalRatio: (intr[0][0] * frame.width) / (intr[1][1] * frame.height),
    principal: [intr[0][2], intr[1][2]],
    skewRatio: (intr[0][1] * frame.width) / (intr[1][1] * frame.height),
    focusDistance,
  };
}

export function cameraToCV(camera: Camera, frame: FrameSize) {
  const matrix = new Matrix4().compose(
    new Vector3(...camera.position),
    new Quaternion(...camera.rotation),
    new Vector3(1, 1, 1),
  );
  const sign = [1, -1, -1, 1];
  const c2w = sign.map((s, i) =>
    sign.map((t, j) => matrix.elements[j * 4 + i] * s * t),
  );
  const k = cameraIntrinsics(camera, frame);
  return {
    intrinsics: [
      [k.fx, k.skew, k.cx],
      [0, k.fy, k.cy],
      [0, 0, 1],
    ],
    c2w,
  };
}

export function cameraAt(path: Camera[], progress: number): Camera {
  const position = clamp(progress, 0, 1) * (path.length - 1);
  const index = Math.floor(position),
    alpha = position - index;
  const a = path[index];
  if (alpha === 0) return a;
  const b = path[index + 1];
  const mix = (x: number, y: number) => x + (y - x) * alpha;
  return {
    position: a.position.map((v, i) => mix(v, b.position[i])) as Vec3,
    rotation: new Quaternion(...a.rotation)
      .slerp(new Quaternion(...b.rotation), alpha)
      .toArray(),
    fov: mix(a.fov, b.fov),
    focalRatio: mix(a.focalRatio, b.focalRatio),
    principal: [
      mix(a.principal[0], b.principal[0]),
      mix(a.principal[1], b.principal[1]),
    ],
    skewRatio: mix(a.skewRatio, b.skewRatio),
    focusDistance: mix(a.focusDistance, b.focusDistance),
  };
}

export function dollyCamera(camera: Camera, distance: number): Camera {
  return {
    ...camera,
    position: add(camera.position, cameraPose(camera).forward, distance),
    focusDistance: camera.focusDistance - distance,
  };
}

// Shared perspective geometry for the scene renderer and the trajectory frusta.
export function projectPoint(
  camera: Camera,
  point: Vec3,
  width: number,
  height: number,
) {
  const pose = cameraPose(camera);
  const relative = add(point, camera.position, -1);
  const depth = dot(relative, pose.forward);
  const k = cameraIntrinsics(camera, { width, height });
  const x = dot(relative, pose.right) / depth;
  const y = -dot(relative, pose.up) / depth;
  return {
    x: (k.fx * x + k.skew * y + k.cx) * width,
    y: (k.fy * y + k.cy) * height,
    depth,
  };
}

export function cameraFrustum(
  camera: Camera,
  distance: number,
  frame: FrameSize,
): Vec3[] {
  const pose = cameraPose(camera);
  const center = add(camera.position, pose.forward, distance);
  const k = cameraIntrinsics(camera, frame);
  return [
    [0, 1],
    [1, 1],
    [1, 0],
    [0, 0],
  ].map(([u, v]) => {
    const y = (v - k.cy) / k.fy;
    const x = (u - k.cx - k.skew * y) / k.fx;
    return add(add(center, pose.right, x * distance), pose.up, -y * distance);
  });
}

export function targetCameraAt(
  source: Camera,
  time: number,
  motion: Motion,
  start: Camera,
): Camera {
  const { preset, direction, amount } = motion;
  const progress = direction * amount * time;
  const pose = cameraPose(start);
  switch (preset) {
    case "original":
      return source;
    case "offset": {
      return {
        ...source,
        position: add(
          source.position,
          cameraPose(source).right,
          direction * amount * source.focusDistance * 0.2,
        ),
      };
    }
    case "fixed":
      return start;
    case "dolly":
      return dollyCamera(start, progress * start.focusDistance * 0.5);
    case "zoom":
      return {
        ...start,
        fov: 2 * Math.atan(Math.tan(start.fov / 2) / 2 ** progress),
      };
    case "truck":
      return {
        ...start,
        position: add(
          start.position,
          pose.right,
          progress * start.focusDistance * 0.4,
        ),
      };
    case "pedestal":
      return {
        ...start,
        position: add(
          start.position,
          [0, 1, 0],
          progress * start.focusDistance * 0.3,
        ),
      };
    case "pan":
      return rotateCamera(start, -(progress * Math.PI) / 3, 0);
    case "tilt":
      return rotateCamera(start, 0, (progress * Math.PI) / 4);
    case "orbit":
      return orbitCamera(start, (progress * Math.PI) / 2, 0);
  }
}
