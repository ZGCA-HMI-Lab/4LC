from __future__ import annotations

from pathlib import Path

import torch

from alc_video.models.video_model.loading import resolve_wan_checkpoint


class WanTextEncoder:
    def __init__(
        self,
        *,
        text_encoder: torch.nn.Module,
        tokenizer,
        device: torch.device,
    ) -> None:
        self.text_encoder = text_encoder.eval()
        self.tokenizer = tokenizer
        self.device = device
        self.text_encoder.requires_grad_(False)

    @classmethod
    def from_local_checkpoint(
        cls,
        checkpoint: str | Path,
        tokenizer: str | Path,
        *,
        device: str | torch.device,
        dtype: torch.dtype = torch.bfloat16,
    ) -> WanTextEncoder:
        from diffsynth.models.model_loader import ModelPool
        from diffsynth.models.wan_video_text_encoder import HuggingfaceTokenizer

        text_path = resolve_wan_checkpoint(
            checkpoint,
            name="Wan text encoder",
        )
        tokenizer_path = Path(tokenizer).expanduser().resolve()
        if not tokenizer_path.is_dir():
            raise FileNotFoundError(
                f"Wan tokenizer directory is missing: {tokenizer_path}"
            )
        resolved_device = torch.device(device)
        pool = ModelPool()
        vram = pool.default_vram_config()
        vram["computation_dtype"] = dtype
        vram["computation_device"] = str(resolved_device)
        pool.auto_load_model(str(text_path), vram_config=vram, clear_parameters=False)
        text_encoder = pool.fetch_model("wan_video_text_encoder")
        if text_encoder is None:
            raise RuntimeError("DiffSynth failed to load the Wan text encoder")
        tokenizer = HuggingfaceTokenizer(
            name=str(tokenizer_path),
            seq_len=512,
            clean="whitespace",
            local_files_only=True,
        )
        return cls(
            text_encoder=text_encoder, tokenizer=tokenizer, device=resolved_device
        )

    @torch.no_grad()
    def _encode_text(self, caption: str, *, allow_empty: bool) -> torch.Tensor:
        if not isinstance(caption, str):
            raise TypeError("caption must be a string")
        if not allow_empty and not caption.strip():
            raise ValueError("caption must be a non-empty string")
        ids, mask = self.tokenizer(caption, return_mask=True, add_special_tokens=True)
        ids = ids.to(self.device)
        mask = mask.to(self.device)
        context = self.text_encoder(ids, mask)
        if tuple(int(v) for v in context.shape) != (1, 512, 4096):
            raise ValueError(
                f"Wan text context shape {tuple(context.shape)} != (1,512,4096)"
            )
        length = int(mask.gt(0).sum().item())
        context[:, length:] = 0
        if not bool(torch.isfinite(context).all()):
            raise ValueError("Wan text context contains non-finite values")
        return context[0].to(device="cpu", dtype=torch.bfloat16).contiguous()

    @torch.no_grad()
    def encode_text(self, caption: str) -> torch.Tensor:
        return self._encode_text(caption, allow_empty=False)

    @torch.no_grad()
    def encode_unconditional(self) -> torch.Tensor:
        return self._encode_text("", allow_empty=True)

    def offload(self) -> None:
        self.text_encoder.to("cpu")
        if self.device.type == "cuda":
            torch.cuda.empty_cache()
