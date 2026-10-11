import { useEffect, useRef, useState } from "react";
import { RotateCcw, X } from "lucide-react";
import { inspectInput } from "./api";
import {
  centeredCrop,
  defaultSamplingFps,
  dragCrop,
  inputEnd,
  inputProblem,
  DEFAULT_FPS,
  OUTPUT_FRAMES,
} from "./input";
import type { Crop, CropHandle, InputSettings, MediaInfo } from "./input";
import type { CacheSettings } from "./input";
import CacheControls from "./CacheControls";

type Props = {
  file: File;
  kind: "image" | "video";
  example?: import("./api.ts").Example;
  onClose: () => void;
  cacheSettings: CacheSettings;
  onConfirm: (settings: InputSettings, cacheSettings: CacheSettings) => void;
};

export default function UploadDialog({
  file,
  kind,
  example,
  onClose,
  onConfirm,
  cacheSettings,
}: Props) {
  const dialog = useRef<HTMLDialogElement>(null);
  const video = useRef<HTMLVideoElement>(null);
  const surface = useRef<HTMLDivElement>(null);
  const drag = useRef<{
    x: number;
    y: number;
    crop: Crop;
    handle: CropHandle;
  } | null>(null);
  const [url, setUrl] = useState<string>();
  const [info, setInfo] = useState<MediaInfo | null>(null);
  const [output, setOutput] = useState<{
    width: number;
    height: number;
  } | null>(null);
  const [crop, setCrop] = useState<Crop | null>(null);
  const [stage, setStage] = useState("Reading input");
  const [error, setError] = useState<string | null>(null);
  const [start, setStart] = useState(0);
  const [fps, setFps] = useState(DEFAULT_FPS);
  const [cache, setCache] = useState<CacheSettings>({ ...(example?.cache ?? cacheSettings) });
  useEffect(() => {
    dialog.current!.showModal();
    const mediaUrl = URL.createObjectURL(file);
    setUrl(mediaUrl);
    const controller = new AbortController();
    void inspectInput(file, kind, setStage, controller.signal, example?.profile)
      .then(({ info, generation }) => {
        if (controller.signal.aborted) return;
        setInfo(info);
        const initialStart = example?.prepare.start ?? 0;
        setStart(initialStart);
        setFps(example?.prepare.sampling_fps ?? defaultSamplingFps(info, initialStart));
        setOutput(generation);
        setCrop(
          example?.prepare.crop ?? centeredCrop(
            info.width,
            info.height,
            generation.width / generation.height,
          ),
        );
      })
      .catch((error: unknown) => {
        if (!controller.signal.aborted)
          setError(
            error instanceof Error ? error.message : "Cannot read this file.",
          );
      });
    return () => {
      controller.abort();
      URL.revokeObjectURL(mediaUrl);
    };
  }, [file, kind, example]);
  const problem =
    info && kind === "video"
      ? inputProblem(info, start, fps) ??
        (example?.time.mode === "freeze" && example.time.at > inputEnd(0, fps)
          ? `The example freezes at ${example.time.at} s, beyond the selected span. Use a longer span or edit its time setting.`
          : null)
      : null;
  const end = inputEnd(start, fps);
  const maxStart =
    info && Number.isFinite(fps) && fps > 0
      ? Math.max(0, info.duration - inputEnd(0, fps))
      : 0;
  function changeStart(value: number) {
    setStart(value);
    if (video.current && Number.isFinite(value))
      video.current.currentTime = value;
  }
  function beginCrop(event: React.PointerEvent, handle: CropHandle) {
    event.stopPropagation();
    event.currentTarget.setPointerCapture(event.pointerId);
    drag.current = { x: event.clientX, y: event.clientY, crop: crop!, handle };
  }
  function moveCrop(event: React.PointerEvent) {
    if (!drag.current || !info || !output) return;
    const bounds = surface.current!.getBoundingClientRect();
    setCrop(
      dragCrop(
        drag.current.crop,
        drag.current.handle,
        ((event.clientX - drag.current.x) * info.width) / bounds.width,
        ((event.clientY - drag.current.y) * info.height) / bounds.height,
        info.width,
        info.height,
        output.width / output.height,
      ),
    );
  }
  return (
    <dialog
      ref={dialog}
      className="upload-dialog"
      aria-labelledby="upload-title"
      onCancel={onClose}
    >
      <header className="upload-header">
        <div>
          <h2 id="upload-title">Prepare input</h2>
          <p>{file.name}</p>
        </div>
        <button
          className="icon-button"
          aria-label="Close upload settings"
          onClick={onClose}
        >
          <X size={19} />
        </button>
      </header>
      <div className="upload-content">
        <div className="crop-stage">
          <div
            className="crop-surface"
            ref={surface}
            style={
              info
                ? {
                    aspectRatio: info.width / info.height,
                    width: `min(100%, calc(var(--crop-height) * ${info.width / info.height}))`,
                  }
                : undefined
            }
          >
            {kind === "video" ? (
              <video
                ref={video}
                src={url}
                muted
                playsInline
                preload="auto"
                onError={() =>
                  setError(
                    "This browser cannot read the video. Upload an H.264 MP4.",
                  )
                }
              />
            ) : (
              <img
                src={url}
                alt="Input crop"
                onError={() =>
                  setError(
                    "This browser cannot read the image. Upload a PNG or JPEG.",
                  )
                }
              />
            )}
            {info && crop && (
              <div
                className="crop-box"
                aria-label="Move crop region"
                style={{
                  left: `${(crop[0] / info.width) * 100}%`,
                  top: `${(crop[1] / info.height) * 100}%`,
                  width: `${(crop[2] / info.width) * 100}%`,
                  height: `${(crop[3] / info.height) * 100}%`,
                }}
                onPointerDown={(event) => beginCrop(event, "move")}
                onPointerMove={moveCrop}
                onPointerUp={() => {
                  drag.current = null;
                }}
                onPointerCancel={() => {
                  drag.current = null;
                }}
              >
                {(["nw", "ne", "sw", "se", "n", "e", "s", "w"] as const).map(
                  (corner) => (
                    <span
                      key={corner}
                      className={`${corner.length === 1 ? "crop-edge" : "crop-handle"} ${corner}`}
                      role="img"
                      aria-label={`Resize crop ${corner}`}
                      onPointerDown={(event) => beginCrop(event, corner)}
                    />
                  ),
                )}
              </div>
            )}
          </div>
          {info && output && (
            <button
              className="crop-reset"
              aria-label="Reset crop"
              onClick={() =>
                setCrop(
                  centeredCrop(
                    info.width,
                    info.height,
                    output.width / output.height,
                  ),
                )
              }
            >
              <RotateCcw size={16} />
            </button>
          )}
        </div>
        {info && kind === "video" && (
          <input
            className="upload-start-slider"
            aria-label="Clip start"
            type="range"
            min="0"
            max={maxStart}
            step="0.01"
            value={Number.isFinite(start) ? start : 0}
            disabled={maxStart === 0}
            onChange={(event) => changeStart(Number(event.target.value))}
          />
        )}
        {info && output && crop ? (
          <div className="upload-comparison">
            <div className="comparison-head">Original</div>
            <div className="comparison-head">Adjust</div>
            <div className="comparison-head">Prepared</div>
            <div>
              <small>Resolution</small>
              <strong>
                {info.width} × {info.height}
              </strong>
            </div>
            <div>
              <small>Crop · drag edges in the preview</small>
              <strong>
                {crop[2]} × {crop[3]}
              </strong>
            </div>
            <div>
              <small>Output resolution</small>
              <strong>
                {output.width} × {output.height}
              </strong>
            </div>
            {kind === "video" && (
              <>
                <div>
                  <small>Frame rate</small>
                  <strong>{info.fps!.toFixed(2)} fps</strong>
                </div>
                <label>
                  <span>Sampling FPS</span>
                  <input
                    type="number"
                    min="0"
                    step="any"
                    value={Number.isFinite(fps) ? fps : ""}
                    onChange={(event) => setFps(event.target.valueAsNumber)}
                  />
                </label>
                <div>
                  <small>Output frame rate</small>
                  <strong>
                    {Number.isFinite(fps) && fps > 0
                      ? `${fps.toFixed(2)} fps`
                      : "—"}
                  </strong>
                </div>
                <div>
                  <small>Duration</small>
                  <strong>{info.duration.toFixed(2)} s</strong>
                </div>
                <label>
                  <span>Start (s)</span>
                  <input
                    type="number"
                    min="0"
                    max={maxStart}
                    step="0.01"
                    value={Number.isFinite(start) ? start : ""}
                    onChange={(event) =>
                      changeStart(event.target.valueAsNumber)
                    }
                  />
                </label>
                <div>
                  <small>Selected range</small>
                  <strong>
                    {Number.isFinite(start) && Number.isFinite(end)
                      ? `${start.toFixed(2)}–${end.toFixed(2)} s`
                      : "—"}
                  </strong>
                </div>
              </>
            )}
          </div>
        ) : (
          !error && (
            <div className="upload-loading" role="status">
              <progress />
              <span>{stage}</span>
            </div>
          )
        )}
        {info && !error && (
          <div className="upload-cache-settings">
            <CacheControls value={cache} onChange={setCache} image={kind === "image"} />
          </div>
        )}
      </div>
      {(error || problem) && (
        <p className="upload-error" role="alert">
          {error || problem}
        </p>
      )}
      <footer className="upload-footer">
        <span>{OUTPUT_FRAMES} frames</span>
        <button className="button" onClick={onClose}>
          Cancel
        </button>
        <button
          className="button primary"
          disabled={!info || !crop || !!error || !!problem}
          onClick={() =>
            onConfirm({
              media_path: info!.media_path,
              start,
              sampling_fps: fps,
              crop: crop!,
            }, cache)
          }
        >
          Prepare Cache
        </button>
      </footer>
    </dialog>
  );
}
