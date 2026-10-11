import { Client, handle_file } from "@gradio/client";
import { decodeScene } from "./scene.ts";
import type { SceneData, SceneDescriptor } from "./scene.ts";
import type { MediaInfo, InputSettings } from "./input.ts";
import type { Motion } from "./camera.ts";

export type Example = {
  id: string;
  name: string;
  kind: "video" | "image";
  url: string;
  prompt: string;
  profile: string;
  mode: "control" | "geometry_i2v";
  prepare: Partial<Omit<InputSettings, "media_path">>;
  camera: Motion;
  time: { mode: "normal" | "freeze" | "reverse"; at: number };
};

export type CacheSummary = {
  source_type: "image" | "video";
  width: number;
  height: number;
  frame_count: number;
  source_duration: number;
  fps: number;
  keyframes: number;
  representations: string[];
  profiles: string[];
  timings: Record<string, number>;
  camera: { intrinsics: number[][][]; c2w: number[][][]; times: number[] };
  scene: SceneDescriptor;
};

export type PreparedCache = CacheSummary & { data: SceneData };

const STAGES: Record<string, string> = {
  "input.decode": "Reading input",
  "input.encode_video": "Preparing input preview",
  "visualize.mask": "Visualizing coverage and motion",
  "comparison.encode_video": "Composing comparison video",
  "geometry.reconstruct": "Reconstructing geometry and motion",
  "geometry.camera": "Estimating the camera",
  "cache.construct": "Encoding the Cache",
  "geometry_and_cache.first_frame": "Reconstructing and encoding the image",
  "cache.transfer_to_cpu": "Preparing the result",
  "scene.pack": "Preparing the scene",
  "cache.transfer_to_gpu": "Preparing the Cache",
  "cache.read": "Reading conditions",
  "visualize.rgb": "Decoding RGB",
  "visualize.depth": "Decoding depth",
  "visualize.vjepa2": "Visualizing V-JEPA",
  "visualize.dinov2": "Visualizing DINOv2",
  "visualize.wm": "Visualizing WorldMirror",
  "preview.encode_video": "Preparing preview videos",
  "conditions.transfer_to_gpu": "Reusing prepared conditions",
  "prompt.encode": "Encoding the prompt",
  "model.transfer_to_gpu": "Loading the generation model onto GPU",
  "model.transfer_to_cpu": "Finishing sampling",
  "diffusion.sample": "Sampling video",
  "video.decode": "Decoding video",
  "video.encode": "Preparing the video download",
};

let connection: Promise<Client> | undefined;

export async function inspectInput(
  file: File,
  kind: "image" | "video",
  onProgress: (stage: string) => void,
  signal: AbortSignal,
  profile?: string,
): Promise<{ info: MediaInfo; generation: { width: number; height: number } }> {
  const response = await fetch(new URL("api/status", window.location.href), {
    signal,
  });
  if (
    !response.ok ||
    !response.headers.get("content-type")?.includes("application/json")
  ) {
    throw new Error(
      "The backend is not connected. Start the Python server to prepare an input.",
    );
  }
  const status = await response.json();
  if (profile && !status.profiles.includes(profile))
    throw new Error(`This example needs model profile ${profile}, which is not loaded.`);
  connection ??= Client.connect(
    new URL("api/gradio", window.location.href).href,
    {
      events: ["data", "status"],
    },
  ).catch((error) => {
    connection = undefined;
    throw error;
  });
  onProgress("Reading media information");
  const info = await submit<MediaInfo>(
    "/inspect_media",
    [handle_file(file), kind],
    onProgress,
    signal,
  );
  return { info, generation: status.generation };
}

export async function prepareCache(
  settings: InputSettings,
  onProgress: (stage: string) => void,
  signal: AbortSignal,
): Promise<PreparedCache> {
  const result = await submit<CacheSummary>(
    "/build_cache",
    [settings],
    onProgress,
    signal,
  );
  onProgress("Loading the scene");
  const scene = await fetch(new URL(result.scene.url, window.location.href), {
    signal,
  });
  if (!scene.ok)
    throw new Error(
      "The scene could not be downloaded. Prepare the Cache again.",
    );
  const data = decodeScene(result.scene, await scene.arrayBuffer());
  signal.throwIfAborted();
  return { ...result, data };
}

async function submit<T>(
  endpoint: string,
  args: unknown[],
  onProgress: (stage: string) => void,
  signal: AbortSignal,
): Promise<T> {
  if (!connection) throw new Error("Prepare a Cache first.");
  const client = await connection;
  signal.throwIfAborted();
  const job = client.submit(endpoint, args);
  const cancel = () => {
    void job.cancel();
  };
  signal.addEventListener("abort", cancel, { once: true });
  let result: T | undefined;
  try {
    for await (const event of job) {
      signal.throwIfAborted();
      if (event.type === "data") result = event.data[0] as T;
      if (event.type !== "status") continue;
      if (event.stage === "error") {
        throw new Error(
          typeof event.message === "string" ? event.message : "Task failed.",
        );
      }
      if (event.stage === "pending") onProgress("Queued");
      for (const progress of event.progress_data ?? []) {
        if (progress.desc) {
          const step =
            progress.desc === "diffusion.sample" && progress.length
              ? ` · ${progress.index}/${progress.length}`
              : "";
          onProgress((STAGES[progress.desc] ?? progress.desc) + step);
        }
      }
    }
    signal.throwIfAborted();
    if (!result) throw new Error("The backend returned no result.");
    return result;
  } finally {
    signal.removeEventListener("abort", cancel);
  }
}

export type ShotPayload = {
  intrinsics: number[][][];
  c2w: number[][][];
  times: number[];
  mode: string;
  profile: string;
};
export type ConditionPreview = {
  videos: Record<string, string>;
  timings: Record<string, number>;
  totalSeconds: number;
};

export function releaseVideos(result: ConditionPreview) {
  Object.values(result.videos).forEach((url) => URL.revokeObjectURL(url));
}

export async function previewConditions(
  shot: ShotPayload,
  onProgress: (stage: string) => void,
  signal: AbortSignal,
): Promise<ConditionPreview> {
  const start = performance.now();
  const result = await submit<{
    videos: Record<string, string>;
    timings: Record<string, number>;
  }>("/preview_conditions", [shot], onProgress, signal);
  onProgress("Loading previews");
  const downloads = await Promise.allSettled(
    Object.entries(result.videos).map(async ([name, url]) => {
      const response = await fetch(new URL(url, window.location.href), {
        signal,
      });
      if (!response.ok) throw new Error(`Could not load the ${name} preview.`);
      return [name, URL.createObjectURL(await response.blob())] as const;
    }),
  );
  const videos: Record<string, string> = {};
  for (const download of downloads)
    if (download.status === "fulfilled")
      videos[download.value[0]] = download.value[1];
  const failed = downloads.find((download) => download.status === "rejected");
  if (failed || signal.aborted) {
    Object.values(videos).forEach((url) => URL.revokeObjectURL(url));
    signal.throwIfAborted();
    throw failed!.reason;
  }
  return {
    ...result,
    videos,
    totalSeconds: (performance.now() - start) / 1000,
  };
}

export type GeneratedVideo = {
  url: string;
  timings: Record<string, number>;
  totalSeconds: number;
};

export async function generateVideo(
  shot: ShotPayload,
  prompt: string,
  onProgress: (stage: string) => void,
  signal: AbortSignal,
): Promise<GeneratedVideo> {
  const start = performance.now();
  const result = await submit<{ url: string; timings: Record<string, number> }>(
    "/generate_video",
    [shot, prompt],
    onProgress,
    signal,
  );
  onProgress("Loading video");
  const response = await fetch(new URL(result.url, window.location.href), {
    signal,
  });
  if (!response.ok) throw new Error("Could not download the generated video.");
  const blob = await response.blob();
  signal.throwIfAborted();
  return {
    ...result,
    url: URL.createObjectURL(blob),
    totalSeconds: (performance.now() - start) / 1000,
  };
}

export type ComparisonVideo = { url: string; archiveUrl: string };

export async function prepareComparison(
  onProgress: (stage: string) => void,
  signal: AbortSignal,
): Promise<ComparisonVideo> {
  const result = await submit<{ url: string; archive_url: string }>(
    "/prepare_comparison",
    [],
    onProgress,
    signal,
  );
  onProgress("Loading comparison");
  const response = await fetch(new URL(result.url, window.location.href), {
    signal,
  });
  if (!response.ok) throw new Error("Could not load the comparison video.");
  const blob = await response.blob();
  signal.throwIfAborted();
  return {
    url: URL.createObjectURL(blob),
    archiveUrl: new URL(result.archive_url, window.location.href).href,
  };
}
