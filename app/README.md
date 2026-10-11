# 4LC App

A single-page workbench for **building a 4D Cache → choosing camera and time controls → optionally previewing conditions → generating a video**. The App imports the repository's Cache, encoders, model loading and sampler. It supports video control, geometry-guided image conditioning and first-frame-only image conditioning. Rollout is outside this Demo's scope.

## Run

Prepare the core project's [environment](https://github.com/ZGCA-HMI-Lab/4LC#installation) and [model assets](https://github.com/ZGCA-HMI-Lab/4LC#models-and-configuration). Use that Python environment for the App. Node.js 22.12 or later is required to build the frontend.

Download the two released adapters from the repository root:

```bash
hf download jianhuizhang/4LC-Video \
  --include '4lc-video-*.safetensors' \
  --local-dir local-assets/models
```

From the same repository root:

```bash
python -m pip install -r app/requirements.txt
npm --prefix app ci
npm --prefix app run build
PYTHONPATH=app python -m server
```

Open `http://127.0.0.1:7860/`. The App uses the current repository by default; `--core-root` can select a separate 4LC checkout.

Both released profiles load by default. They share Wan, its text encoder and codec, and the visual encoder set; each retains its own trained adapter and VACE branch. Choose `--model /path/to/configs/models/flux-vjepa.yaml` to load a single profile. `--cache`, `--device`, `--host` and `--port` select the existing configuration and serving environment. Models load once at server startup; requests transfer the required components to the GPU without downloading weights again.

For development, keep the Python backend running and start Vite separately:

```bash
npm --prefix app run dev -- --host 0.0.0.0 --port 4175
```

Vite proxies `/api` to port 7860. Behind a proxy that serves a URL prefix, supply the frontend's complete URL as `APP_PUBLIC_URL` when starting Vite and pass the backend prefix with `--root-path`. The host allowlist and WebSocket URL come from `APP_PUBLIC_URL`.

## Input and interaction

**Upload media** opens a preparation window with the original, adjustable and prepared parameters side by side below the media preview. Drag the crop or its edges to frame the input; its aspect ratio matches the model output. Invalid media or a selection that extends past the video produces an explicit error and disables confirmation.

Video is resampled to **81 frames** and cropped/resized to the model resolution. By default, the full video is selected and sampling FPS is calculated as `80 / duration`. Change the start and sampling FPS to choose a segment; its end is `start + 80 / sampling_fps`. Sampling uses the nearest source frame, including repeated frames when needed. The scene, condition previews, generated video and comparison exports share this sampling FPS and preserve the selected time span. Cache times start at zero. Images use 16 fps for playback. The backend applies the confirmed crop and sampling on CPU before GPU construction.

The wide workspace uses a 3:2 split: scene, compact camera/time controls and Prompt on the left; conditions and video results on the right. Conditions use two columns of thumbnails, starting with Mask. The browser downloads RGB points and motion once. Three.js renders them locally, with calibrated cameras and the same neighboring-keyframe motion rule as the core Cache. Playback does not request a GPU or repeatedly transfer the point cloud. The underlying Cache stores additional visual representations; the displayed RGB points show geometry and motion. Browser resizing changes display size only. In the scene, ⌘ + wheel on Mac or Ctrl + wheel on Windows moves the camera continuously, with distance-relative sensitivity. It updates the starting view like the existing ＋ / − controls.

**Preview conditions** reads the selected full trajectory and renders VAE/PCA visualizations. PCA bases remain fixed for a Cache. The preview follows the single Shot preview timeline. Changing controls retains the previous preview with a Settings changed marker; Update preview replaces it, and returning to matching settings restores its valid state.

For images, **First-frame only** supplies the original image features at the first frame and leaves later image conditions empty. Moving the scene view does not reproject these features. **Geometry-guided** reprojects the reconstructed static scene along the target camera path; its condition videos follow the camera, and the video model fills uncovered regions and can add motion. After switching modes or adjusting the camera, select **Update preview** to render the new conditions.

The Mask preview combines uncovered, static-hit and dynamic-hit regions after the model's existing spatial/temporal alignment and coverage aggregation. The current published profiles consume coverage; motion color is a readout diagnostic. Clicking a condition card or the generated video opens an independent player; the workspace pauses and keeps its original position.

**Generate video** works with or without a preview; Prompt is optional. Matching conditions are reused. The expanded result switches between Result and Comparison. Prepare comparison uses the saved generated-shot conditions, reusing matching previews or explicitly rendering missing ones. Later edits to controls or previews cannot change its content.

The labeled comparison places Input, Mask and Generated on the first row, followed by the representations: 6 streams in a 2×3 grid or 8 streams in a 3×3 grid with one empty cell. Input is the processed media used for construction; a still image holds for the full duration. All streams have the same frame count and playback FPS. Export offers the generated MP4, a ZIP containing the individual MP4s, and the combined MP4. Cache/tensor export, parameter-file export and import functionality are outside the App's scope. Both operations report their current stage; sampling also reports completed steps. Core readout/sampling time and total browser waiting time are displayed separately. Total time includes queueing, transfers, decoding and result preparation.

The App discovers `example.yaml` files in `examples/<case>/`. Each case contains its input resource and configuration; see the [case format](https://github.com/ZGCA-HMI-Lab/4LC#inference). Selecting a case opens the preparation window with its crop and sampling defaults, then restores its model, Prompt, camera and time settings after construction. `spatial`, `temporal` and `nvs` cases are shown; `rollout` cases are used by the CLI. Refresh after adding local cases. Visitor uploads are not collected.

## Backend

The React frontend and Gradio API share the same origin. Reuse a single Gradio client connection for each session.

| Endpoint | Purpose |
| --- | --- |
| `/api/status` | Loaded profiles, representations, generation dimensions and capabilities |
| Gradio `inspect_media` | Upload and inspect media on CPU |
| Gradio `build_cache` | Apply input settings, reconstruct and encode; return camera, timeline and scene metadata |
| Gradio `preview_conditions` | Read a complete trajectory and return synchronized condition videos |
| Gradio `generate_video` | Generate directly or reuse matching conditions |
| Gradio `prepare_comparison` | Prepare the generated-shot comparison and video-only export |
| Gradio `cache_info` / `release_cache` | Inspect or release the session's Cache |

Each session owns its CPU Cache and completed results. GPU operations are serialized. A failed operation preserves its previous successful result; changing inputs discards the frontend's old scene and results. Cleanup runs on successful and failed model operations.

## Hugging Face Space

Use a **Gradio Space with ZeroGPU**. Copy `README.md`, `space.py`, `requirements.txt` and `packages.txt` from [deploy/huggingface](deploy/huggingface/) into the Space root. The entrypoint checks out `ZGCA-HMI-Lab/4LC` on `main`, builds React, prepares model assets through the deployment helper and launches the same Python service on port 7860.

The Space uses Python 3.10, PyTorch 2.8 / CUDA 12.8 and the matching torch-scatter wheel. `--zerogpu` imports Spaces before model loading, prepares model storage at startup, and requests the full ZeroGPU for construction, preview and generation. Request handlers reuse downloaded files and model storage. Normal GPU serving continues to use the local startup command above.

All required assets have public download sources. The official FLUX.2-klein VAE has the same tensors as the original FLUX.2-dev VAE at the App's bfloat16 precision; its configuration and parameter names match. The App's two adapters come from `jianhuizhang/4LC-Video`.

The public Demo is [4LC-Demo](https://huggingface.co/spaces/jianhuizhang/4LC-Demo). To add examples, commit the case directories, each with its input and `example.yaml`, to the code repository's `examples/` directory, then restart the Space to load the updated repository.
