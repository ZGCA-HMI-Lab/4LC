from __future__ import annotations

import argparse
import sys
from dataclasses import replace
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser(description="Run the 4LC workbench backend.")
    parser.add_argument(
        "--core-root",
        type=Path,
        default=Path(__file__).resolve().parents[2],
        help="4LC code checkout containing alc_video, alc and third_party",
    )
    parser.add_argument(
        "--model",
        type=Path,
        help="Load one model YAML; by default load both released profiles",
    )
    parser.add_argument(
        "--cache",
        type=Path,
        help="Cache YAML; defaults to the core checkout's cache.yaml",
    )
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=7860)
    parser.add_argument(
        "--root-path", default="", help="URL prefix supplied by the reverse proxy"
    )
    parser.add_argument(
        "--zerogpu",
        action="store_true",
        help="Enable HF ZeroGPU execution (requires spaces)",
    )
    args = parser.parse_args()
    root = args.core_root.expanduser().resolve()
    if not (root / "alc_video").is_dir():
        parser.error(
            f"{root} does not contain alc_video; set --core-root to the 4LC repository"
        )
    if not (Path(__file__).resolve().parents[1] / "dist/index.html").is_file():
        parser.error("Build the frontend first: npm --prefix app run build")
    sys.path[:0] = [
        str(root / part)
        for part in (
            "alc/src",
            "third_party/vjepa2",
            "third_party",
            "third_party/vggt",
            ".",
        )
    ]

    # ZeroGPU must intercept CUDA before any model is loaded.
    if args.zerogpu:
        import spaces

        gpu = spaces.GPU
    else:
        gpu = None

    import uvicorn

    from alc_video.inference.config import load_generation_config
    from alc_video.runtime.cli import initialize_cli
    from server.runtime import CacheRuntime
    from server.service import create_app

    initialize_cli()
    paths = (
        [args.model]
        if args.model
        else [
            root / "configs/models/flux-vjepa.yaml",
            root / "configs/models/flux-all.yaml",
        ]
    )
    profiles = {
        path.stem: replace(
            load_generation_config(
                path, cache=args.cache or root / "configs/cache.yaml"
            ),
            device=args.device,
        )
        for path in paths
    }
    config = max(
        profiles.values(), key=lambda profile: len(profile.encoders.representations)
    )
    with CacheRuntime(
        config, profiles=profiles, device=args.device, zerogpu=args.zerogpu
    ) as runtime:
        if args.zerogpu:
            # Register model CUDA storage in ZeroGPU's startup snapshot.
            # Per-operation offloading still bounds activation memory in each worker.
            import torch

            for network in runtime.networks.values():
                network.to(runtime.device, dtype=torch.bfloat16)
            runtime.components.text.text_encoder.to(runtime.device)
            runtime.components.codec.prepare()
            runtime.mirror.reconstructor_module.to(runtime.device)
            for encoder in {
                id(item): item for item in runtime.encoders.values()
            }.values():
                encoder.prepare()
        app = create_app(runtime, gpu=gpu)
        if args.zerogpu:
            # mount_gradio_app does not invoke Spaces' Blocks.launch startup hook.
            from spaces.zero import startup

            startup()
        uvicorn.run(
            app,
            host=args.host,
            port=args.port,
            root_path=args.root_path,
            timeout_graceful_shutdown=5,
            access_log=False,
        )


if __name__ == "__main__":
    main()
