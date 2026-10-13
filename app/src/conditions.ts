import {
  cameraAt,
  cameraToCV,
  targetCameraAt,
  usesInputPath,
} from "./camera.ts";
import type { Camera, FrameSize, Motion } from "./camera";
export type TimePreset = "forward" | "freeze" | "reverse";

export type ConditionSettings = {
  source: string;
  image: boolean;
  frame: FrameSize;
  profile: string;
  imageMode: string;
  startCamera: Camera;
  motion: Motion;
  timePreset: TimePreset;
  freezeTime: number;
  sourceDuration: number;
};

export function generationSceneTime(
  time: number,
  preset: TimePreset,
  freezeTime: number,
): number {
  switch (preset) {
    case "forward":
      return time;
    case "freeze":
      return freezeTime;
    case "reverse":
      return 1 - time;
  }
}

export function sceneSeconds(settings: ConditionSettings, progress: number) {
  return settings.image
    ? 0
    : generationSceneTime(progress, settings.timePreset, settings.freezeTime) *
        settings.sourceDuration;
}

export function shotTrajectory(
  settings: ConditionSettings,
  path: Camera[],
  frames: number,
) {
  const cameras = Array.from({ length: frames }, (_, i) => {
    const progress = frames > 1 ? i / (frames - 1) : 0;
    return cameraToCV(
      targetCameraAt(
        cameraAt(path, progress),
        progress,
        settings.motion,
        settings.startCamera,
      ),
      settings.frame,
    );
  });
  return {
    intrinsics: cameras.map((camera) => camera.intrinsics),
    c2w: cameras.map((camera) => camera.c2w),
    times: cameras.map((_, i) =>
      sceneSeconds(settings, frames > 1 ? i / (frames - 1) : 0),
    ),
    mode: settings.image ? settings.imageMode : "control",
    profile: settings.profile,
  };
}

// Only settings that change conditioning belong here, not playback or prompt.
export function conditionKey(settings: ConditionSettings): string {
  const { motion, startCamera } = settings;
  const adjustable = motion.preset !== "original" && motion.preset !== "fixed";
  return JSON.stringify([
    settings.source,
    settings.image,
    settings.frame.width,
    settings.frame.height,
    settings.profile,
    settings.sourceDuration,
    settings.image ? settings.imageMode : null,
    motion.preset,
    adjustable ? [motion.direction, motion.amount] : null,
    adjustable && !usesInputPath(motion.preset) ? motion.ease : null,
    usesInputPath(motion.preset)
      ? null
      : [
          startCamera.position,
          startCamera.rotation,
          startCamera.fov,
          startCamera.focalRatio,
          startCamera.principal,
          startCamera.skewRatio,
          startCamera.focusDistance,
        ],
    settings.image ? null : settings.timePreset,
    !settings.image && settings.timePreset === "freeze"
      ? settings.freezeTime
      : null,
  ]);
}
