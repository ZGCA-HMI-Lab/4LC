import { useEffect, useMemo, useRef, useState } from "react";
import type { CSSProperties } from "react";
import {
  BookOpen,
  CircleAlert,
  Expand,
  Layers3,
  Minus,
  Plus,
  RotateCcw,
  X,
} from "lucide-react";
import PointCloudScene from "./PointCloudScene";
import ConditionVideo from "./ConditionVideo";
import VideoDialog, { MaskLegend } from "./VideoDialog";
import ExportMenu from "./ExportMenu";
import InputPanel from "./InputPanel";
import UploadDialog from "./UploadDialog";
import { DEFAULT_CACHE_SETTINGS, sameCacheSettings } from "./input";
import type { CacheSettings, InputSettings } from "./input";
import type { UploadedMedia } from "./InputPanel";
import UsageGuide from "./UsageGuide";
import TrajectoryView from "./TrajectoryView";
import CameraPresets from "./CameraPresets";
import PlaybackBar from "./PlaybackBar";
import type { Example } from "./api";
import type { TimePreset } from "./conditions";
import {
  CAMERA_LABELS,
  INITIAL_CAMERA,
  INITIAL_MOTION,
  cameraAt,
  cameraFromCV,
  clamp,
  dollyCamera,
  targetCameraAt,
  usesInputPath,
} from "./camera";
import type { Camera, Motion } from "./camera";
import { conditionKey, sceneSeconds, shotTrajectory } from "./conditions";
import type { ConditionSettings } from "./conditions";
import {
  prepareCache,
  previewConditions,
  releaseVideos,
  generateVideo,
  prepareComparison,
} from "./api";
import type {
  PreparedCache,
  ConditionPreview,
  GeneratedVideo,
  ComparisonVideo,
} from "./api";

type ConditionName =
  | "Mask"
  | "RGB"
  | "Depth"
  | "V-JEPA"
  | "DINOv2"
  | "WorldMirror";
const READOUT_LABELS: Record<ConditionName, string> = {
  Mask: "Coverage & motion",
  RGB: "VAE decode",
  Depth: "VAE decode",
  "V-JEPA": "PCA",
  DINOv2: "PCA",
  WorldMirror: "PCA",
};
const ROLE_LABELS: Record<string, ConditionName> = {
  mask: "Mask",
  rgb: "RGB",
  depth: "Depth",
  vjepa2: "V-JEPA",
  dinov2: "DINOv2",
  wm: "WorldMirror",
};

export default function App() {
  const [examples, setExamples] = useState<Example[]>([]);
  const [startCamera, setStartCamera] = useState<Camera>(INITIAL_CAMERA);
  const [outputTime, setOutputTime] = useState(0);
  const [playing, setPlaying] = useState(false);
  const [motion, setMotion] = useState<Motion>(INITIAL_MOTION);
  const [timePreset, setTimePreset] = useState<TimePreset>("forward");
  const [freezeTime, setFreezeTime] = useState(0.3);
  const [imageMode, setImageMode] = useState("geometry_i2v");
  const [profile, setProfile] = useState("flux-all");
  const [prompt, setPrompt] = useState("");
  const [guideOpen, setGuideOpen] = useState(false);
  const guideTrigger = useRef<HTMLButtonElement>(null);
  const expandedDialog = useRef<HTMLDialogElement>(null);
  const [expanded, setExpanded] = useState<"scene" | "source" | null>(null);
  const [videoView, setVideoView] = useState<
    | {
        kind: "condition";
        name: ConditionName;
        src: string | null;
        settings: ConditionSettings;
      }
    | { kind: "result"; compare: boolean }
    | null
  >(null);
  const generatedPlayer = useRef<HTMLVideoElement>(null);
  const [videoStart, setVideoStart] = useState(0);
  const [comparison, setComparison] = useState<ComparisonVideo | null>(null);
  const [comparisonStage, setComparisonStage] = useState<string | null>(null);
  const [comparisonError, setComparisonError] = useState<string | null>(null);
  const comparisonRequest = useRef<AbortController | null>(null);
  useEffect(
    () => () => {
      comparisonRequest.current?.abort();
    },
    [],
  );
  useEffect(
    () => () => {
      if (comparison) URL.revokeObjectURL(comparison.url);
    },
    [comparison],
  );
  const [uploaded, setUploaded] = useState<UploadedMedia | null>(null);
  const [buildStage, setBuildStage] = useState<string | null>(null);
  const [builtCache, setBuiltCache] = useState<PreparedCache | null>(null);
  const [cacheSettings, setCacheSettings] = useState<CacheSettings>({ ...DEFAULT_CACHE_SETTINGS });
  const [pendingUpload, setPendingUpload] = useState<{
    file: File;
    kind: "image" | "video";
    example?: Example;
  } | null>(null);
  const [inputSettings, setInputSettings] = useState<InputSettings | null>(
    null,
  );
  const exampleRequest = useRef<AbortController | null>(null);
  const buildRequest = useRef<AbortController | null>(null);
  const previewRequest = useRef<AbortController | null>(null);
  const [conditionError, setConditionError] = useState<string | null>(null);
  const [generationError, setGenerationError] = useState<string | null>(null);
  const [conditionStage, setConditionStage] = useState<string | null>(null);
  const [realPreview, setRealPreview] = useState<ConditionPreview | null>(null);
  useEffect(
    () => () => {
      exampleRequest.current?.abort();
      buildRequest.current?.abort();
      previewRequest.current?.abort();
    },
    [],
  );
  useEffect(
    () => () => {
      if (realPreview) releaseVideos(realPreview);
    },
    [realPreview],
  );
  const [videoStage, setVideoStage] = useState<string | null>(null);
  const [generated, setGenerated] = useState<GeneratedVideo | null>(null);
  const generationRequest = useRef<AbortController | null>(null);
  useEffect(
    () => () => {
      generationRequest.current?.abort();
    },
    [],
  );
  useEffect(
    () => () => {
      if (generated) URL.revokeObjectURL(generated.url);
    },
    [generated],
  );
  const [inputError, setInputError] = useState<string | null>(null);
  useEffect(() => {
    const controller = new AbortController();
    fetch(new URL("api/examples", window.location.href), {
      signal: controller.signal,
    })
      .then(async (response) => {
        if (!response.ok) throw new Error("Could not load examples.");
        setExamples(await response.json());
      })
      .catch((error) => {
        if (!controller.signal.aborted) setInputError(error.message);
      });
    return () => controller.abort();
  }, []);
  const image = uploaded?.kind === "image";
  const ready = builtCache !== null;
  const frame = useMemo(
    () =>
      builtCache
        ? { width: builtCache.width, height: builtCache.height }
        : { width: 560, height: 336 },
    [builtCache],
  );
  const frameCount = builtCache?.frame_count ?? 81;
  const duration = builtCache
    ? (builtCache.frame_count - 1) / builtCache.fps
    : 0;
  const sourceDuration = builtCache?.source_duration ?? 0;
  const sourcePath = useMemo(
    () =>
      builtCache
        ? builtCache.camera.c2w.map((pose, i) =>
            cameraFromCV(
              builtCache.camera.intrinsics[i],
              pose,
              frame,
              builtCache.data.focusDistance,
            ),
          )
        : [INITIAL_CAMERA],
    [builtCache, frame],
  );
  const conditionSettings: ConditionSettings = {
    source: inputSettings?.media_path ?? "",
    image,
    frame,
    profile,
    imageMode,
    startCamera,
    motion,
    timePreset,
    freezeTime,
    sourceDuration,
  };
  const [previewedSettings, setPreviewedSettings] =
    useState<ConditionSettings | null>(null);
  const conditionsReady =
    ready &&
    previewedSettings !== null &&
    conditionKey(conditionSettings) === conditionKey(previewedSettings);
  const representations: ConditionName[] = realPreview
    ? Object.keys(realPreview.videos).map((name) => ROLE_LABELS[name])
    : (previewedSettings?.profile ?? profile) === "flux-all"
      ? ["Mask", "RGB", "Depth", "DINOv2", "V-JEPA", "WorldMirror"]
      : ["Mask", "RGB", "Depth", "V-JEPA"];
  const hasPreview = realPreview !== null;
  const sceneMoment = sceneSeconds(conditionSettings, outputTime);
  const viewCamera = useMemo(
    () =>
      targetCameraAt(
        cameraAt(sourcePath, outputTime),
        outputTime,
        motion,
        startCamera,
      ),
    [sourcePath, outputTime, motion, startCamera],
  );
  const sourceRelative = usesInputPath(motion.preset);
  const origin = sourceRelative ? sourcePath[0] : startCamera;
  const path = useMemo(
    () =>
      Array.from({ length: 41 }, (_, i) =>
        targetCameraAt(
          cameraAt(sourcePath, i / 40),
          i / 40,
          motion,
          startCamera,
        ),
      ),
    [sourcePath, motion, startCamera, duration, sourceDuration, image],
  );
  const playbackTime = useRef(outputTime);
  useEffect(() => {
    playbackTime.current = outputTime;
  }, [outputTime]);
  useEffect(() => {
    if (!playing || duration === 0) return;
    let frame: number,
      last = 0;
    const start = performance.now() - playbackTime.current * duration * 1000;
    const tick = (now: number) => {
      if (now - last > 40) {
        const value = ((now - start) % (duration * 1000)) / (duration * 1000);
        setOutputTime(value);
        last = now;
      }
      frame = requestAnimationFrame(tick);
    };
    frame = requestAnimationFrame(tick);
    return () => cancelAnimationFrame(frame);
  }, [playing, duration]);
  useEffect(() => {
    const onKey = (event: KeyboardEvent) => {
      if (event.key === "Escape" && guideOpen && !expanded) {
        setGuideOpen(false);
        guideTrigger.current?.focus();
      }
    };
    window.addEventListener("keydown", onKey);
    return () => window.removeEventListener("keydown", onKey);
  }, [expanded, guideOpen]);
  useEffect(() => {
    if (!expanded) return;
    const dialog = expandedDialog.current!;
    dialog.showModal();
    return () => dialog.close();
  }, [expanded]);
  useEffect(
    () => () => {
      if (uploaded) URL.revokeObjectURL(uploaded.url);
    },
    [uploaded],
  );

  function seek(value: number) {
    setPlaying(false);
    setOutputTime(value);
  }
  function editView(camera: Camera) {
    setPlaying(false);
    setStartCamera(camera);
    if (usesInputPath(motion.preset)) setMotion({ ...motion, preset: "fixed" });
    setOutputTime(0);
  }
  function changeMotion(next: Motion) {
    setMotion(next);
    setPlaying(false);
  }
  function changeTimePreset(next: TimePreset) {
    setTimePreset(next);
    setPlaying(false);
  }
  function changeFreezeTime(value: number) {
    setFreezeTime(value);
    setPlaying(false);
  }
  function resetPreview(first = sourcePath[0], sourceIsImage = image) {
    setPlaying(false);
    setOutputTime(0);
    setStartCamera(first);
    setMotion({
      ...INITIAL_MOTION,
      preset: sourceIsImage ? "fixed" : "original",
    });
    setTimePreset("forward");
    setFreezeTime(0.3);
  }
  function uploadMedia(file: File) {
    exampleRequest.current?.abort();
    const kind = file.type.startsWith("image/")
      ? "image"
      : file.type.startsWith("video/")
        ? "video"
        : null;
    if (!kind) {
      setInputError(
        "Choose an image or video file with a recognized media type.",
      );
      return;
    }
    setPendingUpload({ file, kind });
  }
  function confirmUpload(settings: InputSettings, cache: CacheSettings) {
    if (!pendingUpload) return;
    const example = pendingUpload.example;
    buildRequest.current?.abort();
    previewRequest.current?.abort();
    generationRequest.current?.abort();
    comparisonRequest.current?.abort();
    setComparisonStage(null);
    setComparison(null);
    setComparisonError(null);
    setVideoView(null);
    setVideoStage(null);
    setGenerated(null);
    setConditionStage(null);
    setConditionError(null);
    setGenerationError(null);
    setRealPreview(null);
    setBuildStage(null);
    setBuiltCache(null);
    setCacheSettings(cache);
    setPrompt(pendingUpload.example?.prompt ?? "");
    setUploaded({
      name: example?.name ?? pendingUpload.file.name,
      url: URL.createObjectURL(pendingUpload.file),
      kind: pendingUpload.kind,
      exampleId: pendingUpload.example?.id,
    });
    setInputSettings(settings);
    setPreviewedSettings(null);
    setInputError(null);
    resetPreview(INITIAL_CAMERA, pendingUpload.kind === "image");
    setPendingUpload(null);
    void buildCache(settings, example, cache);
  }
  async function buildCache(settings = inputSettings, example?: Example, cache = cacheSettings, resetShot = true) {
    if (!settings) return;
    const controller = new AbortController();
    buildRequest.current = controller;
    setInputError(null);
    setBuildStage("Connecting");
    try {
      const result = await prepareCache(
        settings,
        cache,
        (stage) => {
          if (!controller.signal.aborted) setBuildStage(stage);
        },
        controller.signal,
      );
      if (!controller.signal.aborted) {
        setPreviewedSettings(null);
        setRealPreview(null);
        setGenerated(null);
        setComparison(null);
        setComparisonError(null);
        setConditionError(null);
        setGenerationError(null);
        setVideoView(null);
        setBuiltCache(result);
        if (!result.profiles.includes(profile)) setProfile(result.profiles[0]);
        if (resetShot) resetPreview(
          cameraFromCV(
            result.camera.intrinsics[0],
            result.camera.c2w[0],
            result,
            result.data.focusDistance,
          ),
          result.source_type === "image",
        );
        if (example) {
          setProfile(example.profile);
          setImageMode(example.mode === "control" ? "geometry_i2v" : example.mode);
          setMotion(example.camera);
          setTimePreset(example.time.mode === "normal" ? "forward" : example.time.mode);
          setFreezeTime(result.source_duration > 0 ? example.time.at / result.source_duration : 0);
        }
      }
    } catch (error) {
      if (!controller.signal.aborted) {
        setInputError(
          error instanceof Error ? error.message : "Cache construction failed.",
        );
      }
    } finally {
      if (!controller.signal.aborted) setBuildStage(null);
    }
  }
  async function previewCurrentConditions() {
    if (!builtCache) return;
    const controller = new AbortController();
    previewRequest.current = controller;
    const captured = structuredClone(conditionSettings);
    setConditionStage("Preparing conditions");
    setConditionError(null);
    try {
      const result = await previewConditions(
        shotTrajectory(captured, sourcePath, frameCount),
        (stage) => {
          if (!controller.signal.aborted) setConditionStage(stage);
        },
        controller.signal,
      );
      if (controller.signal.aborted) {
        releaseVideos(result);
        return;
      }
      setRealPreview(result);
      setPreviewedSettings(captured);
    } catch (error) {
      if (!controller.signal.aborted)
        setConditionError(
          error instanceof Error ? error.message : "Condition preview failed.",
        );
    } finally {
      if (!controller.signal.aborted) setConditionStage(null);
    }
  }
  async function generateCurrentVideo() {
    if (!builtCache) {
      setGenerationError(
        "Upload an image or video and prepare a Cache to generate a video. ",
      );
      return;
    }
    const controller = new AbortController();
    generationRequest.current = controller;
    setVideoStage("Preparing generation");
    setGenerationError(null);
    setPlaying(false);
    try {
      const result = await generateVideo(
        shotTrajectory(conditionSettings, sourcePath, frameCount),
        prompt,
        (stage) => {
          if (!controller.signal.aborted) setVideoStage(stage);
        },
        controller.signal,
      );
      if (controller.signal.aborted) {
        URL.revokeObjectURL(result.url);
        return;
      }
      setGenerated(result);
      setComparison(null);
      setComparisonError(null);
    } catch (error) {
      if (!controller.signal.aborted)
        setGenerationError(
          error instanceof Error ? error.message : "Video generation failed.",
        );
    } finally {
      if (!controller.signal.aborted) setVideoStage(null);
    }
  }
  async function prepareCurrentComparison() {
    const controller = new AbortController();
    comparisonRequest.current = controller;
    setComparisonStage("Preparing comparison");
    setComparisonError(null);
    try {
      const result = await prepareComparison((stage) => {
        if (!controller.signal.aborted) setComparisonStage(stage);
      }, controller.signal);
      if (controller.signal.aborted) {
        URL.revokeObjectURL(result.url);
        return;
      }
      setComparison(result);
    } catch (error) {
      if (!controller.signal.aborted)
        setComparisonError(
          error instanceof Error
            ? error.message
            : "Comparison preparation failed.",
        );
    } finally {
      if (!controller.signal.aborted) setComparisonStage(null);
    }
  }
  function openResult(compare = false) {
    setPlaying(false);
    setVideoStart(generatedPlayer.current!.currentTime);
    generatedPlayer.current!.pause();
    setVideoView({ kind: "result", compare });
  }
  function openCondition(name: ConditionName) {
    setPlaying(false);
    setVideoStart(outputTime * duration);
    const role = Object.keys(ROLE_LABELS).find(
      (role) => ROLE_LABELS[role] === name,
    )!;
    setVideoView({
      kind: "condition",
      name,
      src: realPreview ? realPreview.videos[role] : null,
      settings: previewedSettings!,
    });
  }

  async function chooseExample(example: Example) {
    exampleRequest.current?.abort();
    const controller = new AbortController();
    exampleRequest.current = controller;
    try {
      const response = await fetch(new URL(example.url, window.location.href), {
        signal: controller.signal,
      });
      if (!response.ok) throw new Error("Could not load this example.");
      const blob = await response.blob();
      controller.signal.throwIfAborted();
      setPendingUpload({
        file: new File([blob], example.id.split("/").at(-1)!, {
          type: blob.type,
        }),
        kind: example.kind,
        example,
      });
    } catch (error) {
      if (!controller.signal.aborted)
        setInputError(
          error instanceof Error
            ? error.message
            : "Could not load this example.",
        );
    }
  }
  const emptyMessage =
    !ready || buildStage ? (
      <div className="media-empty">
        <Layers3 size={27} />
        <span role="status">
          {buildStage ?? (builtCache ? "Cache prepared" : "Prepare Cache")}
        </span>
        {buildStage && <progress aria-label="Cache construction progress" />}
      </div>
    ) : null;
  return (
    <div className="app-shell" id="top">
      <header className="app-toolbar">
        <a className="brand" href="#top" aria-label="4LC workspace">
          <img src="./4lc.svg" alt="4LC" width="60" height="28" />
        </a>
        <div className="toolbar-links">
          <button
            ref={guideTrigger}
            className="guide-trigger"
            aria-expanded={guideOpen}
            aria-controls="usage-guide"
            onClick={() => setGuideOpen(!guideOpen)}
          >
            <BookOpen size={18} aria-hidden="true" />
            Usage guide
          </button>
          <a
            className="github-link"
            href="https://github.com/ZGCA-HMI-Lab/4LC"
            target="_blank"
            rel="noopener noreferrer"
            aria-label="GitHub"
          >
            <img src="./github.svg" alt="" width="24" height="24" />
          </a>
        </div>
      </header>
      <div className="workbench-layout">
        <InputPanel
          examples={examples}
          uploaded={uploaded}
          onExample={(example) => void chooseExample(example)}
          onUpload={uploadMedia}
          onPreview={() => setExpanded("source")}
          onPrepare={() => void buildCache(inputSettings, undefined, cacheSettings, false)}
          preparing={buildStage !== null}
          busy={
            conditionStage !== null ||
            videoStage !== null ||
            comparisonStage !== null
          }
          prepared={builtCache !== null}
          settings={cacheSettings}
          onSettingsChange={setCacheSettings}
          changed={builtCache !== null && !image && !sameCacheSettings(cacheSettings, builtCache.cache)}
        />
        <main id="workspace">
          {inputError && (
            <div className="notice" role="alert">
              {inputError}
            </div>
          )}
          <div className="workspace-grid">
            <div className="pane scene-column">
              <section className="scene-pane" aria-label="4D Scene">
                <header className="pane-heading">
                  <h1>4D Scene</h1>
                  <button
                    className="icon-button"
                    aria-label="Expand scene"
                    disabled={!ready}
                    onClick={() => setExpanded("scene")}
                  >
                    <Expand size={16} />
                  </button>
                </header>
                <div
                  className="scene-viewport"
                  style={{
                    aspectRatio: frame.width / frame.height,
                  }}
                >
                  {builtCache && (
                    <PointCloudScene
                      data={builtCache.data}
                      camera={viewCamera}
                      frame={frame}
                      time={sceneMoment}
                      onCameraChange={editView}
                      onError={setInputError}
                    />
                  )}
                  {emptyMessage}
                  <div className="viewport-tools">
                    <button
                      aria-label="Move view closer"
                      disabled={!ready}
                      onClick={() =>
                        editView(
                          dollyCamera(
                            viewCamera,
                            Math.min(0.7, viewCamera.focusDistance * 0.2),
                          ),
                        )
                      }
                    >
                      <Plus size={18} />
                    </button>
                    <button
                      aria-label="Move view farther"
                      disabled={!ready}
                      onClick={() => editView(dollyCamera(viewCamera, -0.7))}
                    >
                      <Minus size={18} />
                    </button>
                    <button
                      aria-label="Reset view"
                      disabled={!ready}
                      onClick={() => resetPreview()}
                    >
                      <RotateCcw size={16} />
                    </button>
                  </div>
                </div>
              </section>
              <PlaybackBar
                label="Shot preview"
                time={outputTime}
                sceneTime={
                  !image && timePreset !== "forward" ? sceneMoment : undefined
                }
                duration={duration}
                frameCount={frameCount}
                playing={playing}
                disabled={!ready || duration === 0}
                onTime={seek}
                onPlay={() => setPlaying(!playing)}
              />
            </div>
            <aside className="pane shot-workspace">
              <div className="controls-panel" aria-label="Generation controls">
                <CameraPresets
                  image={image}
                  motion={motion}
                  disabled={!ready}
                  onChange={changeMotion}
                />
                <section
                  className="control-section"
                  aria-label="Time mode settings"
                >
                  <h2>{image ? "Conditioning" : "Time mode"}</h2>
                  {image ? (
                    <select
                      aria-label="Image conditioning mode"
                      disabled={!ready}
                      value={imageMode}
                      onChange={(event) => {
                        setImageMode(event.target.value);
                        setPlaying(false);
                      }}
                    >
                      <option value="geometry_i2v">Geometry-guided</option>
                      <option value="i2v">First-frame only</option>
                    </select>
                  ) : (
                    <>
                      <div
                        className="segmented"
                        role="group"
                        aria-label="Time mode"
                      >
                        {(
                          [
                            { value: "forward", label: "Normal" },
                            { value: "freeze", label: "Freeze" },
                            { value: "reverse", label: "Reverse" },
                          ] as const
                        ).map((item) => (
                          <button
                            key={item.value}
                            className={
                              timePreset === item.value ? "active" : ""
                            }
                            aria-pressed={timePreset === item.value}
                            disabled={!ready}
                            onClick={() => {
                              changeTimePreset(item.value);
                            }}
                          >
                            {item.label}
                          </button>
                        ))}
                      </div>
                      {timePreset === "freeze" && (
                        <div className="numeric-control">
                          <label htmlFor="freeze-time">Freeze at</label>
                          <label className="time-input">
                            <input
                              id="freeze-time"
                              type="number"
                              disabled={!ready}
                              min="0"
                              max={sourceDuration}
                              step={sourceDuration / 80}
                              value={Number(
                                (freezeTime * sourceDuration).toFixed(2),
                              )}
                              onChange={(event) => {
                                const value = event.currentTarget.valueAsNumber;
                                if (
                                  Number.isFinite(value) &&
                                  sourceDuration > 0
                                )
                                  changeFreezeTime(
                                    clamp(value / sourceDuration, 0, 1),
                                  );
                              }}
                            />
                            <span>s</span>
                          </label>
                          <input
                            type="range"
                            min="0"
                            max="80"
                            step="1"
                            value={Math.round(freezeTime * 80)}
                            aria-label="Freeze scene time"
                            disabled={!ready}
                            onChange={(event) =>
                              changeFreezeTime(Number(event.target.value) / 80)
                            }
                          />
                        </div>
                      )}
                    </>
                  )}
                </section>
              </div>
              <section
                className="pane trajectory-pane"
                aria-label="Camera trajectory"
              >
                <header className="pane-heading">
                  <h2>Camera trajectory</h2>
                  <span className="pane-meta">
                    {CAMERA_LABELS[motion.preset]}
                  </span>
                </header>
                {ready && (
                  <TrajectoryView
                    path={path}
                    referencePath={
                      sourceRelative && !image ? sourcePath : undefined
                    }
                    current={viewCamera}
                    origin={origin}
                    frame={frame}
                    disabled={!ready}
                  />
                )}
              </section>
            </aside>
            <aside
              className="pane conditions-panel"
              aria-label="Condition previews"
            >
              <header className="conditions-heading">
                <div className="condition-title">
                  <h2>Conditions</h2>
                </div>
                <select
                  className="profile-select"
                  aria-label="Representation set"
                  value={profile}
                  onChange={(event) => {
                    setProfile(event.target.value);
                  }}
                >
                  {(builtCache?.profiles ?? ["flux-all", "flux-vjepa"]).map(
                    (name) => (
                      <option key={name} value={name}>
                        {name === "flux-vjepa"
                          ? "FLUX + V-JEPA"
                          : name === "flux-all"
                            ? "FLUX + All"
                            : name}
                      </option>
                    ),
                  )}
                </select>
              </header>
              <div className="conditions-body">
                {conditionStage && (
                  <div className="media-empty task-status" role="status">
                    <span>{conditionStage}</span>
                    <progress aria-label="Condition preview progress" />
                  </div>
                )}
                <div className="readout-stack">
                  {representations.map((name) => (
                    <button
                      key={name}
                      className="readout-tile"
                      disabled={
                        !hasPreview ||
                        (name === "Mask" && !realPreview) ||
                        conditionStage !== null
                      }
                      aria-label={
                        hasPreview
                          ? "Expand " + name + " readout"
                          : name + " condition not previewed"
                      }
                      onClick={() => openCondition(name)}
                    >
                      <div className="readout-title">
                        <span>
                          {name}
                          <small>{READOUT_LABELS[name]}</small>
                        </span>
                        {hasPreview && <Expand size={14} />}
                      </div>
                      <div
                        className="readout-image"
                        style={
                          {
                            "--frame-ratio": frame.width / frame.height,
                          } as CSSProperties
                        }
                      >
                        {realPreview && builtCache && (
                          <ConditionVideo
                            src={
                              realPreview.videos[
                                Object.keys(ROLE_LABELS).find(
                                  (role) => ROLE_LABELS[role] === name,
                                )!
                              ]
                            }
                            progress={outputTime}
                            frames={frameCount}
                            fps={builtCache.fps}
                            label={name + " condition preview"}
                          />
                        )}
                        {!hasPreview && !conditionStage && (
                          <span className="readout-empty idle">
                            Not previewed
                          </span>
                        )}
                        {name === "Mask" && realPreview && <MaskLegend />}
                      </div>
                    </button>
                  ))}
                </div>
              </div>
              <footer
                className="conditions-actions"
                aria-label="Condition preview actions"
              >
                {conditionError && (
                  <p className="action-error" role="alert">
                    {conditionError}
                  </p>
                )}
                <div className="preview-action-row">
                  {previewedSettings && !conditionsReady && (
                    <span className="condition-status" role="status">
                      <CircleAlert size={14} />
                      Settings changed
                    </span>
                  )}
                  <button
                    className="button primary"
                    disabled={
                      !ready ||
                      conditionsReady ||
                      conditionStage !== null ||
                      videoStage !== null ||
                      comparisonStage !== null ||
                      buildStage !== null
                    }
                    onClick={previewCurrentConditions}
                  >
                    {conditionStage
                      ? "Previewing…"
                      : previewedSettings && !conditionsReady
                        ? "Update preview"
                        : "Preview conditions"}
                  </button>
                </div>
                {realPreview && (
                  <output className="task-timing">
                    {realPreview.timings["cache.read"] === undefined
                      ? "Reused conditions"
                      : `Readout ${(realPreview.timings["cache.read"] * 1000).toFixed(0)} ms`}{" "}
                    · Total {realPreview.totalSeconds.toFixed(1)} s
                  </output>
                )}
              </footer>
            </aside>
            <section
              className={`pane generation-panel${generated ? "" : " is-empty"}`}
              aria-label="Generated video"
            >
              <header className="pane-heading">
                <h2>Video</h2>
                <button
                  className="icon-button"
                  aria-label="Expand generated video"
                  disabled={!generated || videoStage !== null}
                  onClick={() => openResult()}
                >
                  <Expand size={16} />
                </button>
              </header>
              <div
                className="output-placeholder"
                style={
                  generated
                    ? { aspectRatio: frame.width / frame.height }
                    : undefined
                }
              >
                {generated && (
                  <button
                    className="output-preview"
                    aria-label="Open generated video"
                    disabled={videoStage !== null}
                    onClick={() => openResult()}
                  >
                    <video
                      ref={generatedPlayer}
                      src={generated.url}
                      muted
                      playsInline
                      preload="auto"
                      aria-label="Generated video result"
                    />
                  </button>
                )}
                {!generated && !videoStage && !comparisonStage && (
                  <span className="output-empty">
                    Generated video appears here
                  </span>
                )}
                {(videoStage || comparisonStage) && (
                  <div className="media-empty task-status" role="status">
                    <span>{videoStage ?? comparisonStage}</span>
                    <progress aria-label="Video generation progress" />
                  </div>
                )}
              </div>
              {generated && (
                <footer className="generation-results">
                  <output className="task-timing">
                    Sampling {generated.timings["diffusion.sample"].toFixed(1)}{" "}
                    s · Total {generated.totalSeconds.toFixed(1)} s
                  </output>
                  <ExportMenu
                    video={generated.url}
                    comparison={comparison}
                    onComparison={() => openResult(true)}
                    disabled={videoStage !== null || comparisonStage !== null}
                  />
                </footer>
              )}
            </section>
            <section className="pane generation-form" aria-label="Video prompt">
              {generationError && (
                <p className="action-error" role="alert">
                  {generationError}
                </p>
              )}
              <label htmlFor="prompt">Prompt (optional)</label>
              <textarea
                id="prompt"
                value={prompt}
                rows={3}
                onChange={(event) => setPrompt(event.target.value)}
              />
              <div className="generation-actions">
                <button
                  className="button primary"
                  disabled={
                    !ready ||
                    videoStage !== null ||
                    conditionStage !== null ||
                    comparisonStage !== null ||
                    buildStage !== null
                  }
                  onClick={() => void generateCurrentVideo()}
                >
                  {videoStage ? "Generating…" : "Generate video"}
                </button>
              </div>
            </section>
          </div>
        </main>
      </div>
      {pendingUpload && (
        <UploadDialog
          key={pendingUpload.file.name + pendingUpload.file.lastModified}
          file={pendingUpload.file}
          kind={pendingUpload.kind}
          example={pendingUpload.example}
          cacheSettings={cacheSettings}
          onClose={() => setPendingUpload(null)}
          onConfirm={confirmUpload}
        />
      )}
      {videoView && (videoView.kind === "condition" || generated) && (
        <VideoDialog
          title={
            videoView.kind === "condition"
              ? `${videoView.name} condition`
              : "Generated video"
          }
          src={videoView.kind === "condition" ? videoView.src : generated!.url}
          initialTime={videoStart}
          mask={videoView.kind === "condition" && videoView.name === "Mask"}
          onClose={() => setVideoView(null)}
          exportControl={
            videoView.kind === "result" && generated ? (
              <ExportMenu
                video={generated.url}
                comparison={comparison}
                onComparison={() =>
                  setVideoView({ kind: "result", compare: true })
                }
                disabled={comparisonStage !== null}
              />
            ) : undefined
          }
          comparison={
            videoView.kind === "result"
              ? {
                  src: comparison?.url ?? null,
                  stage: comparisonStage,
                  error: comparisonError,
                  onPrepare: () => void prepareCurrentComparison(),
                  selected: videoView.compare,
                  onSelect: (compare) =>
                    setVideoView({ kind: "result", compare }),
                }
              : undefined
          }
        />
      )}
      {expanded && (
        <dialog
          ref={expandedDialog}
          className="expanded-pane"
          onClose={() => setExpanded(null)}
          aria-label={expanded === "source" ? "Input preview" : "4D Scene"}
          onClick={(event) => {
            if (event.target !== event.currentTarget) return;
            const bounds = event.currentTarget.getBoundingClientRect();
            if (
              event.clientX < bounds.left ||
              event.clientX > bounds.right ||
              event.clientY < bounds.top ||
              event.clientY > bounds.bottom
            )
              expandedDialog.current?.close();
          }}
        >
          <header className="pane-heading">
            <h2>{expanded === "source" ? uploaded?.name : "4D Scene"}</h2>
            <button
              className="icon-button"
              aria-label="Close expanded view"
              onClick={() => expandedDialog.current?.close()}
            >
              <X size={18} />
            </button>
          </header>
          <div className="expanded-media">
            {expanded === "source" && uploaded ? (
              uploaded.kind === "image" ? (
                <img src={uploaded.url} alt="Uploaded input" />
              ) : (
                <video src={uploaded.url} controls />
              )
            ) : expanded === "scene" && builtCache ? (
              <PointCloudScene
                data={builtCache.data}
                camera={viewCamera}
                frame={frame}
                time={sceneMoment}
                onCameraChange={editView}
                onError={setInputError}
              />
            ) : null}
          </div>
        </dialog>
      )}
      {guideOpen && (
        <UsageGuide
          onClose={() => {
            setGuideOpen(false);
            guideTrigger.current?.focus();
          }}
        />
      )}
    </div>
  );
}
