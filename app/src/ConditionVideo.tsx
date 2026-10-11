import { useEffect, useRef } from "react";

export default function ConditionVideo({
  src,
  progress,
  frames,
  fps,
  label,
}: {
  src: string;
  progress: number;
  frames: number;
  fps: number;
  label: string;
}) {
  const video = useRef<HTMLVideoElement>(null);
  // Seek to the center of the requested frame, avoiding timestamp-rounding boundaries.
  const target = useRef(0);
  target.current = (Math.round(progress * (frames - 1)) + 0.5) / fps;
  const synchronize = () => {
    const element = video.current;
    if (
      element &&
      element.readyState >= 1 &&
      !element.seeking &&
      Math.abs(element.currentTime - target.current) > 0.25 / fps
    )
      element.currentTime = target.current;
  };
  useEffect(synchronize, [progress, frames, fps, src]);
  return (
    <video
      ref={video}
      src={src}
      muted
      playsInline
      preload="auto"
      aria-label={label}
      onLoadedData={synchronize}
      onSeeked={synchronize}
    />
  );
}
