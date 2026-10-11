<h2 align="center">4LC: 4D Latent Cache for Efficient Conditioning in Dynamic World Modeling</h2>

<div align="center">

Jianhui Zhang<sup>1,2</sup> &nbsp; Zuyao Lin<sup>3,2</sup> &nbsp; Pengxin Xu<sup>4,2</sup> &nbsp; Jingdong Wang<sup>2,✉</sup> &nbsp; Xingyu Chen<sup>2,✉</sup>

<sub><sup>1</sup> SJTU · <sup>2</sup> ZGCA · <sup>3</sup> CASIA · <sup>4</sup> HIT &emsp; ✉&nbsp;Equal&nbsp;advising</sub>

<!-- TODO: replace XXXX.XXXXX with the final arXiv identifier throughout this file. -->
<p>
  <a href="https://arxiv.org/abs/XXXX.XXXXX"><img src="https://img.shields.io/badge/arXiv-Paper-B31B1B?style=flat&amp;logo=arxiv&amp;logoColor=white&amp;labelColor=454545" height="22" alt="Paper"></a>
  <a href="https://zgca-hmi-lab.github.io/4LC/"><img src="https://img.shields.io/badge/Project-Page-0F766E?style=flat&amp;labelColor=454545" height="22" alt="Project page"></a>
  <a href="https://huggingface.co/jianhuizhang/4LC-Video"><img src="https://img.shields.io/badge/Hugging_Face-Models-FFD21E?style=flat&amp;logo=huggingface&amp;logoColor=FFD21E&amp;labelColor=454545" height="22" alt="Models"></a>
  <a href="https://huggingface.co/spaces/jianhuizhang/4LC-Demo"><img src="https://img.shields.io/badge/Spaces-Demo-0284C7?style=flat&amp;logo=huggingface&amp;logoColor=FFD21E&amp;labelColor=454545" height="22" alt="Demo"></a>
  <a href="LICENSE"><img src="https://img.shields.io/badge/License-MIT-64748B?style=flat&amp;labelColor=454545" height="22" alt="License: MIT"></a>
</p>

<!-- Main visual: replace this teaser with the project video when available. -->
<p><img src="assets/readme/teaser.png" width="100%" alt="4LC: construction, spatiotemporal readout and video generation"></p>

<p><strong>4LC caches visual representations of static and dynamic content for reuse across times and viewpoints.</strong></p>

[Latent Cache](#latent-cache) · [Installation](#installation) · [Models & Config](#models-and-configuration) · [Inference](#inference) · [Training](#training)

</div>

<!-- News: add a heading and dated announcements when releases are available. -->

## Latent Cache

The standalone `latent-cache` library provides **Construction, Readout and Composition** with caller-supplied encoders, geometry and motion. It requires Python 3.10+ and PyTorch.

```bash
git clone https://github.com/ZGCA-HMI-Lab/4LC.git
cd 4LC
python -m pip install -e ./alc
```

**Encode once, query across views and time.** Given an encoder and prepared observations, configure the Cache and bind each representation to its input stream. See the [complete example](alc/README.md#quick-start) for input preparation.

```python
from latent_cache import LatentCache, LatentCacheConfig

config = LatentCacheConfig(
    image_size=image_size, native_resolution=True,
)
cache = LatentCache(config, {"rgb": (rgb_encoder, "rgb")})
entry = cache.construct(**observations)

# Reuse the observation along another camera path.
views = cache.read(entry, query_times, query_intr, camera_path)

# Keep scene time fixed while the camera moves.
frozen_times = query_times[:, :1].expand_as(query_times)
frozen = cache.read(entry, frozen_times, query_intr, camera_path)

# Add another observation and fill gaps from static history.
cache.construct(**more_observations)
history = cache.compose(query_times, query_intr, camera_path)
background = history.readouts
```

`LatentCacheConfig` controls motion classification and projection. The video application reads the same fields from [cache.yaml](configs/cache.yaml): select it with inference `--cache`, or the training YAML's `cache` reference. See [Cache configuration](alc/README.md#configuration), [input and encoder conventions](alc/README.md#inputs-and-encoders), and [editing](alc/README.md#editing).

## 4LC-Video

### Installation

**Environment and PyTorch** — Linux, Python 3.10, PyTorch 2.6.0 / CUDA 12.4. Video inference and training have been tested on an A100 80 GB.

Run from the repository root:

```bash
conda create -n 4lc python=3.10 -y
conda activate 4lc
python -m pip install torch==2.6.0 torchvision==0.21.0 \
  --index-url https://download.pytorch.org/whl/cu124
```

**Compute extensions**

```bash
python -m pip install torch-scatter \
  -f https://data.pyg.org/whl/torch-2.6.0+cu124.html
python -m pip install gsplat==1.5.3
```

**Core library and application dependencies**

```bash
python -m pip install -e ./alc
python -m pip install -r requirements.txt \
  diffusers==0.37.1 deepspeed==0.16.7
source scripts/env.sh
```

Run `source scripts/env.sh` from the repository root in each new shell to use the bundled DiffSynth, WorldMirror and encoder code.

The [interactive App](app/README.md) supports uploading media, browsing the Cache, previewing conditions and generating videos. It runs locally on a GPU or on Hugging Face ZeroGPU.

### Models and Configuration

Both profiles use **Wan2.1-T2V-14B**. Choose a profile and its matching learned weights:

| Profile | Cached representations | Weights |
| :--- | :--- | :--- |
| [flux-vjepa](configs/models/flux-vjepa.yaml) (default) | RGB / depth / V-JEPA 2.1 | [Download](https://huggingface.co/jianhuizhang/4LC-Video/resolve/main/4lc-video-flux-vjepa.safetensors) |
| [flux-all](configs/models/flux-all.yaml) | RGB / depth / DINOv2 / V-JEPA 2.1 / WorldMirror | [Download](https://huggingface.co/jianhuizhang/4LC-Video/resolve/main/4lc-video-flux-all.safetensors) |

Download the weights to the default location:

```bash
hf download jianhuizhang/4LC-Video \
  --include "4lc-video-*.safetensors" \
  --local-dir local-assets/models
```

`--model` selects the profile; `--weights` overrides the learned-weight path. The supplied video encoders operate at **336 × 560**.

<details>
<summary><b>Download base models and place encoder assets</b></summary>

| Asset | Role |
| :--- | :--- |
| [Wan2.1-T2V-14B](https://huggingface.co/Wan-AI/Wan2.1-T2V-14B/tree/main) | Video backbone, VAE, text encoder and tokenizer |
| [FLUX.2 VAE](https://huggingface.co/black-forest-labs/FLUX.2-dev/tree/main/vae) | RGB and depth features |
| [V-JEPA 2.1 ViT-L/16](https://dl.fbaipublicfiles.com/vjepa2/vjepa2_1_vitl_dist_vitG_384.pt) | Video features |
| [NeoVerse WorldMirror](https://huggingface.co/Yuppie1204/NeoVerse) | Geometry; additional features in `flux-all` |
| [DINOv2-large](https://huggingface.co/facebook/dinov2-large) | Features in `flux-all` only |
| [LightX2V rank-64 LoRA](https://huggingface.co/lightx2v/Wan2.1-T2V-14B-StepDistill-CfgDistill-Lightx2v/tree/main/loras) | Four-step inference and rollout |

Place the files under these paths, or change their locations in the model and encoder YAML files. For FLUX.2, copy the contents of its `vae/` folder.

```text
local-assets/models/
├── Wan2.1-T2V-14B/
│   ├── diffusion_pytorch_model*.safetensors
│   ├── Wan2.1_VAE.pth
│   ├── models_t5_umt5-xxl-enc-bf16.pth
│   └── google/umt5-xxl/
├── FLUX.2-VAE/
│   ├── config.json
│   └── diffusion_pytorch_model.safetensors
├── vjepa2/vjepa2_1_vitl_dist_vitG_384.pt
├── NeoVerse/reconstructor.ckpt
├── dinov2/                       # flux-all only: HF config and weights
└── lightx2v/
    └── Wan21_T2V_14B_lightx2v_cfg_step_distill_lora_rank64.safetensors
```

</details>

**Where to configure each part**

| Configuration | Controls |
| :--- | :--- |
| [cache.yaml](configs/cache.yaml) | Motion classification, projection and readout batching |
| [encoders.yaml](configs/encoders.yaml) | Encoder backends, asset paths and encoding batch sizes |
| [Model profiles](configs/models) | Ordered representation bindings, adapter and control-branch structure |
| [Training profiles](configs/train-flux-vjepa.yaml) | Data, optimizer, updates, checkpoints and shared configuration references |
| [augmentation.yaml](configs/augmentation.yaml) | Training augmentation and unconditional context |

Keep the representation bindings and model structure consistent with the chosen checkpoint. YAML paths are relative to the declaring file; CLI paths are relative to the working directory. Training `--set` overrides apply after model and Cache references are loaded.

### Inference

**Choose a conditioning mode**

| `--mode` | Source | Conditioning | Default camera |
| :--- | :--- | :--- | :--- |
| `control` | Video | Read video features at target times and viewpoints | Original input path |
| `i2v` | Image | Image features guide the first frame; later frames are generated | Fixed input view |
| `geometry_i2v` | Image | Reproject the reconstructed image into target views for NVS | Fixed input view |

The input type and default mode are inferred from the file. Both image modes use WorldMirror and the visual encoders. Prompt is optional. LightX2V uses four sampling steps by default; `--no-lightx2v` disables it and `--steps` changes the step count.

**Run a case**

Each directory under `examples/` contains its input and an `example.yaml`. The CLI and [App](app/README.md) load the same input preparation, model and shot settings:

```bash
python scripts/infer.py --example examples/spatial-control --output outputs/control.mp4
python scripts/infer.py --example examples/nvs --output outputs/nvs.mp4
```

**Control your own video**

The full input span is sampled to **81 frames**, keeping its duration, and 21 keyframes are encoded by default. `--start` selects the beginning in seconds; `--sampling-fps` sets the spacing, so the end is `start + 80 / sampling_fps`. Without a sampling rate, the remaining video is used. `--crop X Y W H` selects source pixels; otherwise a centered crop matches the model's aspect ratio. The shared input processor then resizes to the configured resolution.

```bash
python scripts/infer.py --input path/to/video.mp4 \
  --camera offset --direction left --amount 0.3 \
  --time freeze --freeze-at 2 \
  --output outputs/control.mp4
```

`--time normal`, `freeze` and `reverse` control scene time independently of the camera. Freeze times are measured from the start of the prepared clip. Video playback retains the selected input span; images default to 16 fps.

**Generate from an image**

Choose `geometry_i2v` to guide each target view with reconstructed geometry. `i2v` uses the original image features only at the first frame; changing the camera does not reproject that first-frame condition.

```bash
python scripts/infer.py --input path/to/image.png --mode geometry_i2v \
  --camera orbit --direction right --amount 0.3 \
  --prompt "A smooth camera move through the scene." \
  --output outputs/nvs.mp4
```

<details>
<summary><b>Camera presets</b></summary>

CLI presets start from the input's first estimated camera. The App also lets you choose a starting view interactively. `--amount` ranges from 0 to 1, with default 0.4. Translation and orbit radius use the first view's valid median depth, denoted D below.

| Preset | Directions | Effect at amount 1 |
| :--- | :--- | :--- |
| `original` | — | Follow the reconstructed video camera path |
| `offset` | Left / right | Shift the whole input path by 0.2 D along each camera's right axis |
| `fixed` | — | Keep the starting camera and lens |
| `dolly` | In / out | Move 0.5 D forward or backward, keeping the lens |
| `zoom` | In / out | Double or halve focal length, keeping the camera pose |
| `truck` | Left / right | Move 0.4 D sideways without rotating |
| `pedestal` | Up / down | Move 0.3 D along world vertical |
| `pan` | Left / right | Turn 60° in place |
| `tilt` | Up / down | Tilt 45° in place |
| `orbit` | Left / right | Orbit 90° around a pivot D ahead of the starting camera |

The default direction is right, in or up for the selected preset. `original` and `offset` require video input. CLI and browser implementations use the same camera conventions and preset definitions.

</details>

**Rollout with Cache history**

Rollout starts from **one image** and generates 81-frame windows. Each window shares its first frame with the previous one; N windows produce **80 × N + 1 frames**. Cache history is enabled by default and uses `geometry_i2v`.

```bash
# One outgoing window followed by the exact return path: 161 frames.
python scripts/rollout.py --input path/to/image.png \
  --camera dolly --direction in --amount 0.3 --revisit \
  --output outputs/revisit.mp4

# Multi-action exploration defined by a case.
python scripts/rollout.py --example examples/exploration \
  --output outputs/exploration.mp4
```

For a single action, `--chunks` sets its number of windows. **Amount is the total change over that action**, so more windows extend its duration. A multi-action case carries each final camera pose and lens into the next action. `revisit: true` appends the exact reversed camera path and doubles the window count. The initial scene scale remains fixed throughout the route.

Each window samples 21 keyframes and writes up to 20 new static Cache keyframes. `--order`, `--max-entries` and `--coverage-threshold` control history reuse. `--no-use-cache` generates from each preceding final frame, with `i2v` or `geometry_i2v`. Supply one Prompt for the whole route, or repeat `--prompt` once per window. Cache `target_chunk_size` controls readout batching.

<details>
<summary><b>Example directory and configuration</b></summary>

```text
examples/
└── spatial-control/
    ├── input.mp4
    └── example.yaml
```

A video case:

```yaml
task: spatial                    # spatial, temporal, nvs or rollout
input: input.mp4                  # relative to this file
model: flux-vjepa                 # profile name, or a relative model YAML path
prompt: A continuous shot with natural motion.
prepare:                         # optional; omitted values use full-span / centered-crop defaults
  start: 0
  sampling_fps: 16
camera: {preset: offset, direction: left, amount: 0.3}
time: {mode: normal}              # freeze uses at: <seconds>; reverse reads time backwards
```

For NVS, use `task: nvs`, an image input and a camera move or `fixed`. Images have no source time controls. A rollout case uses `actions` in place of `camera` and `time`:

```yaml
task: rollout
input: input.png
model: flux-vjepa
prompt: A smooth exploration of the scene.
actions:
  - {preset: dolly, direction: in, amount: 0.3, chunks: 1}
  - {preset: pan, direction: right, amount: 0.3, chunks: 1}
  - {preset: dolly, direction: in, amount: 0.3, chunks: 1}
revisit: true
```

`prepare.crop` accepts `[x, y, width, height]` in source pixels. Explicit CLI options override case settings; multi-action routes are edited in the case's `actions`. Unknown configuration fields fail with their field paths. The App discovers single-shot case directories and restores their settings; rollout cases remain CLI examples.

</details>

**Outputs and reruns**

Each run saves `.mp4`, `.json` (effective configuration and frame count) and `.log` files with the same basename. For rollout, repeat the command with `--resume` to skip matching completed output or restart incomplete output from the first frame. Use a new output path for another ordinary inference run.

### Training

**Prepare data and text context**

Place [SpatialVID-HQ](https://github.com/NJU-3DV/SpatialVID) metadata, videos and captions under `data/SpatialVID/`, or set `data.root` in the training YAML:

```text
data/SpatialVID/
├── data/train/SpatialVID_HQ_metadata.csv
└── SpatialVid/HQ/
    ├── videos/<group>/<id>.mp4
    └── annotations/<group>/<id>/caption.json
```

Captions use `SceneDescription`. Training samples 81 frames at 16 fps and computes geometry and features online. After downloading Wan, generate the empty-prompt context once:

```bash
python scripts/prepare_unconditional_context.py
```

This writes the file below, already referenced by [augmentation.yaml](configs/augmentation.yaml). A custom `--output` must also be set in `condition.prompt_drop.unconditional_context`.

```text
local-assets/augmentation/wan21-unconditional-context.safetensors
```

**Choose a profile and train**

| Profile | Output directory |
| :--- | :--- |
| [train-flux-vjepa.yaml](configs/train-flux-vjepa.yaml) | `outputs/training-flux-vjepa/` |
| [train-flux-all.yaml](configs/train-flux-all.yaml) | `outputs/training-flux-all/` |

One-GPU training with CPU optimizer offload:

```bash
python scripts/train.py --config configs/train-flux-vjepa.yaml \
  --gpus 1 --set training.zero_optimizer_offload=cpu
```

Set `--gpus` for multi-GPU training. Override fields with repeated `--set`, such as `--set optimizer.lr=1.0e-5`. `training.max_updates` is the total target update count, including completed updates.

<details>
<summary><b>Initialize, resume and export</b></summary>

`checkpoint.initial_weights` initializes the learned adapter and control branch. `checkpoint.resume_from` restores model, optimizer, scheduler, RNG and data-reader state:

```bash
python scripts/train.py --config configs/train-flux-vjepa.yaml \
  --gpus 1 --set training.zero_optimizer_offload=cpu \
  --set checkpoint.resume_from=outputs/training-flux-vjepa/latest
```

Export the learned weights for inference:

```bash
python scripts/export_model.py outputs/training-flux-vjepa/latest \
  --output outputs/releases/4lc-video-flux-vjepa.safetensors
```

Pass the exported file to inference with `--weights`, using the same model profile as training.

</details>

## Citation

If you find our library, code, or paper useful in your research, please cite our work:

<!-- TODO: replace XXXX.XXXXX with the final arXiv identifier. -->
```bibtex
@article{zhang2026fourlc,
  title   = {{4LC}: {4D} Latent Cache for Efficient Conditioning
             in Dynamic World Modeling},
  author  = {Zhang, Jianhui and Lin, Zuyao and Xu, Pengxin
             and Wang, Jingdong and Chen, Xingyu},
  journal = {arXiv preprint arXiv:XXXX.XXXXX},
  year    = {2026}
}
```

## Acknowledgements

We thank the authors of [Wan](https://github.com/Wan-Video/Wan2.1), [DiffSynth-Studio](https://github.com/modelscope/DiffSynth-Studio), [NeoVerse](https://github.com/IamCreateAI/NeoVerse), [FLUX.2](https://github.com/black-forest-labs/flux2), [V-JEPA](https://github.com/facebookresearch/vjepa2), [DINOv2](https://github.com/facebookresearch/dinov2), [LightX2V](https://github.com/ModelTC/LightX2V) and [SpatialVID](https://github.com/NJU-3DV/SpatialVID) for their open-source contributions.
