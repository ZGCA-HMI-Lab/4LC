import { CAMERA_LABELS, CUSTOM_PRESETS } from "./camera";
import type { CameraPreset, Motion } from "./camera";

type Props = {
  image: boolean;
  motion: Motion;
  disabled: boolean;
  onChange: (motion: Motion) => void;
};

export default function CameraPresets({
  image,
  motion,
  disabled,
  onChange,
}: Props) {
  const adjustable = motion.preset !== "original" && motion.preset !== "fixed";
  const directions =
    motion.preset === "dolly" || motion.preset === "zoom"
      ? [
          { value: 1, label: "In" },
          { value: -1, label: "Out" },
        ]
      : motion.preset === "pedestal" || motion.preset === "tilt"
        ? [
            { value: 1, label: "Up" },
            { value: -1, label: "Down" },
          ]
        : [
            { value: -1, label: "Left" },
            { value: 1, label: "Right" },
          ];
  return (
    <section className="control-section" aria-label="Camera controls">
      <h2>Camera</h2>
      <label className="control-field">
        <select
          aria-label="Camera motion"
          value={motion.preset}
          disabled={disabled}
          onChange={(event) =>
            onChange({
              ...motion,
              preset: event.target.value as CameraPreset,
              direction: 1,
            })
          }
        >
          {!image && (
            <optgroup label="Based on input">
              <option value="original">{CAMERA_LABELS.original}</option>
              <option value="offset">{CAMERA_LABELS.offset}</option>
            </optgroup>
          )}
          <optgroup label="From the scene view">
            {CUSTOM_PRESETS.map((preset) => (
              <option value={preset} key={preset}>
                {CAMERA_LABELS[preset]}
              </option>
            ))}
          </optgroup>
        </select>
      </label>
      {adjustable && (
        <>
          <div className="control-field">
            <span>Direction</span>
            <div
              className="segmented"
              role="group"
              aria-label="Motion direction"
            >
              {directions.map(({ value, label }) => (
                <button
                  key={value}
                  className={motion.direction === value ? "active" : ""}
                  aria-pressed={motion.direction === value}
                  disabled={disabled}
                  onClick={() =>
                    onChange({ ...motion, direction: value as -1 | 1 })
                  }
                >
                  {label}
                </button>
              ))}
            </div>
          </div>
          <label className="control-field">
            <span>
              {motion.preset === "offset" ? "Offset" : "Amount"}
              <output>{Math.round(motion.amount * 100)}%</output>
            </span>
            <input
              type="range"
              aria-label="Motion amount"
              min="0"
              max="100"
              value={Math.round(motion.amount * 100)}
              disabled={disabled}
              onChange={(event) =>
                onChange({
                  ...motion,
                  amount: Number(event.target.value) / 100,
                })
              }
            />
          </label>
        </>
      )}
    </section>
  );
}
