from __future__ import annotations

from dataclasses import dataclass, field, replace
from pathlib import Path
from time import perf_counter

import gradio as gr
from fastapi import FastAPI, HTTPException, Response
from fastapi.staticfiles import StaticFiles

from alc_video.inference.media import inspect_media
from server.comparison import Comparison, compose_comparison
from server.conditioning import Shot
from server.examples import discover_examples
from server.media import encode_video
from server.runtime import BuiltCache, CacheRuntime
from server.scene import SceneAsset, export_scene
from server.visualization import render_mask


@dataclass(frozen=True)
class ConditionResult:
    shot: Shot
    readouts: dict


@dataclass(frozen=True)
class Preview:
    condition: ConditionResult
    videos: dict[str, bytes]
    timings: dict[str, float]

    def summary(self, session_id):
        return {
            "videos": {
                name: f"api/conditions/{session_id}/{name}" for name in self.videos
            },
            "timings": self.timings,
        }


@dataclass(frozen=True)
class GeneratedVideo:
    data: bytes
    timings: dict[str, float]
    condition: ConditionResult
    preview: Preview | None
    comparison: Comparison | None = None

    def summary(self, session_id):
        return {"url": f"api/video/{session_id}", "timings": self.timings}


@dataclass(frozen=True)
class Session:
    cache: BuiltCache
    scene: SceneAsset
    input_video: bytes
    pca: dict = field(default_factory=dict)
    preview: Preview | None = None
    conditions: ConditionResult | None = None
    video: GeneratedVideo | None = None

    def summary(self, session_id: str):
        return {
            **self.cache.summary(),
            "scene": {**self.scene.metadata, "url": f"api/scene/{session_id}"},
        }


def create_app(runtime: CacheRuntime, *, gpu=None) -> FastAPI:
    sessions: dict[str, Session] = {}

    def release(session_id):
        sessions.pop(session_id, None)

    def construct(prepared, progress):
        return runtime.construct(prepared, lambda stage: progress(None, desc=stage))

    compute = (
        gpu(duration=60, size="xlarge")(construct) if gpu is not None else construct
    )

    def render(built, shot, pca, progress, existing):
        return runtime.preview(
            built, shot, pca, lambda stage: progress(None, desc=stage), existing
        )

    render_conditions = (
        gpu(duration=60, size="xlarge")(render) if gpu is not None else render
    )

    def generate(built, shot, prompt, existing, progress):
        return runtime.generate(
            built,
            shot,
            prompt,
            existing,
            lambda stage: progress(None, desc=stage),
            lambda step, total: progress((step, total), desc="diffusion.sample"),
        )

    generate_frames = (
        gpu(duration=90, size="xlarge")(generate) if gpu is not None else generate
    )
    tracker = gr.Progress()

    def inspect_input(media, source_type, inspected):
        try:
            info = inspect_media(media, source_type)
        except (ValueError, OSError) as error:
            raise gr.Error(f"Cannot read this input: {error}") from error
        return info.summary(), {**inspected, info.path: info}

    def build_cache(settings, inspected, request: gr.Request, progress=tracker):
        if settings["media_path"] not in inspected:
            raise gr.Error("Upload an image or video before preparing the Cache.")
        start = perf_counter()
        progress(None, desc="input.decode")
        prepared = runtime.prepare(inspected[settings["media_path"]], settings)
        result = compute(prepared, progress)
        progress(None, desc="scene.pack")
        packed_at = perf_counter()
        scene = export_scene(result.entry, result.colors)
        result.timings["scene.pack"] = perf_counter() - packed_at
        session_id = request.session_hash
        progress(None, desc="input.encode_video")
        input_start = perf_counter()
        input_frames = (
            prepared.frames.permute(0, 2, 3, 1).mul(255).round().clamp(0, 255).byte()
        )
        if prepared.kind == "image":
            input_frames = input_frames.expand(result.frame_count, -1, -1, -1)
        input_video = encode_video(input_frames, result.fps)
        result.timings["input.encode_video"] = perf_counter() - input_start
        result.timings["processing.total"] = perf_counter() - start
        session = Session(result, scene, input_video)
        sessions[session_id] = session
        return {
            **session.summary(session_id),
            "profiles": list(runtime.profiles),
        }, session_id

    def cache_info(state: str | None):
        if state not in sessions:
            raise gr.Error("Prepare a Cache first.")
        return {**sessions[state].summary(state), "profiles": list(runtime.profiles)}

    def make_preview(session, shot, progress, existing):
        start = perf_counter()
        readouts, frames, pca, timings = render_conditions(
            session.cache, shot, session.pca, progress, existing
        )
        progress(None, desc="visualize.mask")
        mask_start = perf_counter()
        frames = {
            "mask": render_mask(
                readouts, session.cache.image_size, session.cache.frame_count
            ),
            **frames,
        }
        timings["visualize.mask"] = perf_counter() - mask_start
        progress(None, desc="preview.encode_video")
        encode_start = perf_counter()
        videos = {
            name: encode_video(video, session.cache.fps)
            for name, video in frames.items()
        }
        timings["preview.encode_video"] = perf_counter() - encode_start
        timings["processing.total"] = perf_counter() - start
        return Preview(ConditionResult(shot, readouts), videos, timings), pca

    def matching_preview(session, shot):
        candidates = (session.preview, session.video.preview if session.video else None)
        return next(
            (
                preview
                for preview in candidates
                if preview and preview.condition.shot.matches(shot)
            ),
            None,
        )

    def preview_conditions(value, state, progress=tracker):
        if state not in sessions:
            raise gr.Error("Prepare a Cache first.")
        session = sessions[state]
        shot = Shot.parse(value, session.cache)
        if shot.profile not in runtime.profiles:
            raise gr.Error(f"Model profile {shot.profile!r} is not loaded.")
        retained = matching_preview(session, shot)
        if retained is not None:
            sessions[state] = replace(
                session, preview=retained, conditions=retained.condition
            )
            return retained.summary(state)
        existing = (
            session.conditions.readouts
            if session.conditions and session.conditions.shot.matches(shot)
            else None
        )
        preview, pca = make_preview(session, shot, progress, existing)
        sessions[state] = replace(
            session, preview=preview, pca=pca, conditions=preview.condition
        )
        return preview.summary(state)

    def generate_video(value, prompt, state, progress=tracker):
        if state not in sessions:
            raise gr.Error("Prepare a Cache first.")
        start = perf_counter()
        session = sessions[state]
        shot = Shot.parse(value, session.cache)
        if shot.profile not in runtime.profiles:
            raise gr.Error(f"Model profile {shot.profile!r} is not loaded.")
        if not isinstance(prompt, str):
            raise gr.Error("Prompt must be text.")
        candidates = (
            session.conditions,
            session.preview.condition if session.preview else None,
        )
        existing = next(
            (item for item in candidates if item and item.shot.matches(shot)), None
        )
        readouts = existing.readouts if existing else None
        frames, conditions, timings = generate_frames(
            session.cache, shot, prompt, readouts, progress
        )
        progress(None, desc="video.encode")
        encode_start = perf_counter()
        data = encode_video(frames, session.cache.fps)
        timings["video.encode"] = perf_counter() - encode_start
        timings["processing.total"] = perf_counter() - start
        matching_preview = (
            session.preview
            if session.preview and session.preview.condition.shot.matches(shot)
            else None
        )
        result = GeneratedVideo(
            data, timings, ConditionResult(shot, conditions), matching_preview
        )
        sessions[state] = replace(
            session, video=result, conditions=ConditionResult(shot, conditions)
        )
        return result.summary(state)

    def prepare_comparison(state, progress=tracker):
        if state not in sessions or sessions[state].video is None:
            raise gr.Error("Generate a video first.")
        session = sessions[state]
        generated = session.video
        if generated.comparison is None:
            preview = matching_preview(session, generated.condition.shot)
            pca = session.pca
            if preview is None:
                preview, pca = make_preview(
                    session,
                    generated.condition.shot,
                    progress,
                    generated.condition.readouts,
                )
            progress(None, desc="comparison.encode_video")
            comparison = compose_comparison(
                {
                    "input": session.input_video,
                    "generated": generated.data,
                    **preview.videos,
                },
                session.cache.fps,
                session.cache.image_size,
                session.cache.frame_count,
            )
            sessions[state] = replace(
                session,
                pca=pca,
                video=replace(generated, preview=preview, comparison=comparison),
            )
        return {"url": f"api/comparison/{state}", "archive_url": f"api/export/{state}"}

    with gr.Blocks(title="4LC API", analytics_enabled=False) as api:
        state = gr.State(None, delete_callback=release)
        inspected = gr.State({})
        settings = gr.JSON(label="Input settings")
        media = gr.File(type="filepath")
        kind = gr.Radio(["video", "image"], value="video")
        result = gr.JSON()
        shot = gr.JSON(label="Camera and time trajectory")
        prompt = gr.Textbox(label="Prompt")
        gr.Button("Inspect input").click(
            inspect_input,
            inputs=[media, kind, inspected],
            outputs=[result, inspected],
            api_name="inspect_media",
        )
        gr.Button("Prepare Cache").click(
            build_cache,
            inputs=[settings, inspected],
            outputs=[result, state],
            api_name="build_cache",
            concurrency_id="gpu",
        )
        gr.Button("Cache info").click(
            cache_info,
            inputs=[state],
            outputs=[result],
            api_name="cache_info",
        )
        gr.Button("Preview conditions").click(
            preview_conditions,
            inputs=[shot, state],
            outputs=[result],
            api_name="preview_conditions",
            concurrency_id="gpu",
        )
        gr.Button("Generate video").click(
            generate_video,
            inputs=[shot, prompt, state],
            outputs=[result],
            api_name="generate_video",
            concurrency_id="gpu",
        )
        gr.Button("Prepare comparison").click(
            prepare_comparison,
            inputs=[state],
            outputs=[result],
            api_name="prepare_comparison",
            concurrency_id="gpu",
        )
        gr.Button("Release Cache").click(
            release,
            inputs=[state],
            outputs=[state],
            api_name="release_cache",
            concurrency_id="gpu",
        )
    api.queue()
    app = FastAPI()

    examples = Path(__file__).resolve().parents[2] / "examples"

    @app.get("/api/examples")
    def list_examples():
        return discover_examples(examples)

    app.mount("/api/example-files", StaticFiles(directory=examples), name="examples")

    @app.get("/api/status")
    def status():
        return {
            "ready": True,
            "capabilities": [
                "construct_cache",
                "scene",
                "preview_conditions",
                "generate_video",
                "prepare_comparison",
            ],
            "profiles": list(runtime.profiles),
            "representations": [
                name for name, _ in runtime.config.encoders.representations
            ],
            "generation": {
                "height": runtime.config.cache.image_size[0],
                "width": runtime.config.cache.image_size[1],
            },
        }

    @app.get("/api/scene/{session_id}")
    def scene_data(session_id: str):
        if session_id not in sessions:
            raise HTTPException(
                404, "This scene is no longer available. Prepare the Cache again."
            )
        return Response(
            sessions[session_id].scene.data,
            media_type="application/octet-stream",
            headers={"Cache-Control": "no-store"},
        )

    @app.get("/api/conditions/{session_id}/{name}")
    def condition_video(session_id: str, name: str):
        session = sessions.get(session_id)
        if (
            session is None
            or session.preview is None
            or name not in session.preview.videos
        ):
            raise HTTPException(404, "This condition preview is no longer available.")
        return Response(
            session.preview.videos[name],
            media_type="video/mp4",
            headers={"Cache-Control": "no-store"},
        )

    @app.get("/api/video/{session_id}")
    def generated_video(session_id: str):
        session = sessions.get(session_id)
        if session is None or session.video is None:
            raise HTTPException(404, "This generated video is no longer available.")
        return Response(
            session.video.data,
            media_type="video/mp4",
            headers={"Cache-Control": "no-store"},
        )

    def require_comparison(session_id):
        session = sessions.get(session_id)
        if session is None or session.video is None or session.video.comparison is None:
            raise HTTPException(404, "Prepare the comparison video first.")
        return session.video.comparison

    @app.get("/api/comparison/{session_id}")
    def comparison_video(session_id: str):
        return Response(
            require_comparison(session_id).video,
            media_type="video/mp4",
            headers={"Cache-Control": "no-store"},
        )

    @app.get("/api/export/{session_id}")
    def export_videos(session_id: str):
        return Response(
            require_comparison(session_id).archive(),
            media_type="application/zip",
            headers={
                "Content-Disposition": 'attachment; filename="4lc-videos.zip"',
                "Cache-Control": "no-store",
            },
        )

    app = gr.mount_gradio_app(app, api, path="/api/gradio", show_error=True)
    app.mount(
        "/",
        StaticFiles(directory=Path(__file__).resolve().parents[1] / "dist", html=True),
        name="frontend",
    )
    return app
