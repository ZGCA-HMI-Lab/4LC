import { useId } from "react";
import { RotateCcw } from "lucide-react";
import { DEFAULT_CACHE_SETTINGS, KEYFRAME_COUNTS } from "./input";
import type { CacheSettings } from "./input";

type Props = {
  value: CacheSettings;
  onChange: (value: CacheSettings) => void;
  disabled?: boolean;
  image?: boolean;
};

export default function CacheControls({ value, onChange, disabled, image }: Props) {
  const id = useId();
  return (
    <fieldset className="cache-controls" aria-label="Cache construction settings" disabled={disabled || image}>
      <div className="cache-control">
        <label className="cache-control-label" htmlFor={`${id}-motion`}>
          <span>Motion threshold</span>
          <output htmlFor={`${id}-motion`}>{value.motion_threshold.toFixed(2)}</output>
        </label>
        <div className="motion-control">
          <span aria-hidden="true">0</span>
          <input
            id={`${id}-motion`}
            type="range"
            min={0}
            max={1}
            step={0.01}
            value={value.motion_threshold}
            onChange={(event) => onChange({ ...value, motion_threshold: Number(event.target.value) })}
          />
          <span aria-hidden="true">1</span>
          <button
            type="button"
            className="icon-button"
            aria-label="Reset motion threshold to 0.1"
            disabled={value.motion_threshold === DEFAULT_CACHE_SETTINGS.motion_threshold}
            onClick={() => onChange({
              ...value,
              motion_threshold: DEFAULT_CACHE_SETTINGS.motion_threshold,
            })}
          >
            <RotateCcw size={15} aria-hidden="true" />
          </button>
        </div>
      </div>
      <div className="cache-control">
        <div className="cache-control-label" id={`${id}-keyframes`}>
          <span>Keyframes</span>
          {image ? <output>1 · Image</output> : (
            <span className="keyframe-recommendation">21 recommended</span>
          )}
        </div>
        {!image && (
          <div className="keyframe-stops" role="radiogroup" aria-labelledby={`${id}-keyframes`}>
            {KEYFRAME_COUNTS.map((count) => (
              <label className="keyframe-stop" key={count}>
                <input
                  type="radio"
                  name={`${id}-keyframes`}
                  value={count}
                  checked={value.num_keyframes === count}
                  onChange={() => onChange({ ...value, num_keyframes: count })}
                />
                <span className="keyframe-dot" />
                <span>{count}</span>
              </label>
            ))}
          </div>
        )}
      </div>
    </fieldset>
  );
}
