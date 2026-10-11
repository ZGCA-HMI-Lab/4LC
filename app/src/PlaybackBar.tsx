import {
  ChevronLeft,
  ChevronRight,
  Pause,
  Play,
  RotateCcw,
} from "lucide-react";
import { clamp } from "./camera";

type Props = {
  label: string;
  time: number;
  sceneTime?: number;
  duration: number;
  frameCount: number;
  playing: boolean;
  disabled: boolean;
  onTime: (time: number) => void;
  onPlay: () => void;
};

export default function PlaybackBar({
  label,
  time,
  sceneTime,
  duration,
  frameCount,
  playing,
  disabled,
  onTime,
  onPlay,
}: Props) {
  return (
    <section className="playback-bar" aria-label={label + " playback"}>
      <div className="playback-controls">
        <button
          className="play-button"
          aria-label={(playing ? "Pause " : "Play ") + label.toLowerCase()}
          onClick={onPlay}
          disabled={disabled}
        >
          {playing ? (
            <Pause size={14} fill="currentColor" />
          ) : (
            <Play size={14} fill="currentColor" />
          )}
        </button>
        <button
          className="icon-button"
          aria-label={"Previous " + label.toLowerCase() + " frame"}
          onClick={() =>
            onTime(clamp(time - 1 / Math.max(1, frameCount - 1), 0, 1))
          }
          disabled={disabled}
        >
          <ChevronLeft size={15} />
        </button>
        <button
          className="icon-button"
          aria-label={"Next " + label.toLowerCase() + " frame"}
          onClick={() =>
            onTime(clamp(time + 1 / Math.max(1, frameCount - 1), 0, 1))
          }
          disabled={disabled}
        >
          <ChevronRight size={15} />
        </button>
        <button
          className="icon-button"
          aria-label="Return to shot start"
          onClick={() => onTime(0)}
          disabled={disabled}
        >
          <RotateCcw size={14} />
        </button>
      </div>
      <div className="playback-track">
        <div className="playback-labels">
          <strong>{label}</strong>
          {sceneTime !== undefined && <span>Scene {sceneTime.toFixed(2)} s</span>}
        </div>
        <input
          type="range"
          min="0"
          max={frameCount - 1}
          step="1"
          value={Math.round(time * (frameCount - 1))}
          aria-label={label}
          disabled={disabled}
          onChange={(event) =>
            onTime(Number(event.target.value) / Math.max(1, frameCount - 1))
          }
        />
      </div>
      <div className="playback-time">
        <label className="time-input">
          <span className="sr-only">{label + " in seconds"}</span>
          <input
            type="number"
            min="0"
            max={duration}
            step={duration / Math.max(1, frameCount - 1)}
            value={Number((time * duration).toFixed(2))}
            disabled={disabled}
            onChange={(event) => {
              const value = event.currentTarget.valueAsNumber;
              if (Number.isFinite(value))
                onTime(clamp(duration ? value / duration : 0, 0, 1));
            }}
          />
          <span>s</span>
        </label>
        <span className="playback-duration">/ {duration.toFixed(2)} s</span>
      </div>
    </section>
  );
}
