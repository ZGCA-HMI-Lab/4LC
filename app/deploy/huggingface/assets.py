"""Download public model assets once, before the serving process starts."""

from pathlib import Path
from urllib.request import urlretrieve

from huggingface_hub import hf_hub_download, snapshot_download


def prepare_assets(core: Path) -> None:
    models = core / "local-assets/models"
    models.mkdir(parents=True, exist_ok=True)

    def snapshot(repo, destination, patterns):
        root = Path(snapshot_download(repo, allow_patterns=patterns))
        for path in root.rglob("*"):
            if path.is_file():
                target = models / destination / path.relative_to(root)
                target.parent.mkdir(parents=True, exist_ok=True)
                target.unlink(missing_ok=True)
                target.hardlink_to(path.resolve())

    def asset(repo, filename, destination):
        path = Path(hf_hub_download(repo, filename))
        target = models / destination
        target.parent.mkdir(parents=True, exist_ok=True)
        target.unlink(missing_ok=True)
        target.hardlink_to(path.resolve())

    snapshot(
        "Wan-AI/Wan2.1-T2V-14B",
        "Wan2.1-T2V-14B",
        [
            "diffusion_pytorch_model*.safetensors",
            "Wan2.1_VAE.pth",
            "models_t5_umt5-xxl-enc-bf16.pth",
            "google/umt5-xxl/*",
        ],
    )
    for name in ("config.json", "diffusion_pytorch_model.safetensors"):
        asset("black-forest-labs/FLUX.2-klein-4B", f"vae/{name}", f"FLUX.2-VAE/{name}")
    asset("Yuppie1204/NeoVerse", "reconstructor.ckpt", "NeoVerse/reconstructor.ckpt")
    snapshot("facebook/dinov2-large", "dinov2", ["config.json", "model.safetensors"])
    lora = "Wan21_T2V_14B_lightx2v_cfg_step_distill_lora_rank64.safetensors"
    asset(
        "lightx2v/Wan2.1-T2V-14B-StepDistill-CfgDistill-Lightx2v",
        f"loras/{lora}",
        f"lightx2v/{lora}",
    )
    for profile in ("flux-vjepa", "flux-all"):
        name = f"4lc-video-{profile}.safetensors"
        asset("jianhuizhang/4LC-Video", name, name)
    vjepa = models / "vjepa2/vjepa2_1_vitl_dist_vitG_384.pt"
    if not vjepa.exists():
        vjepa.parent.mkdir(parents=True, exist_ok=True)
        partial = vjepa.with_suffix(".download")
        urlretrieve(
            "https://dl.fbaipublicfiles.com/vjepa2/vjepa2_1_vitl_dist_vitG_384.pt",
            partial,
        )
        partial.rename(vjepa)
