from __future__ import annotations

import argparse
import multiprocessing
import os
import sys
from pathlib import Path

from alc_video.utils.logging import (
    LOG,
    LOG_LEVELS,
)


def main(argv=None):
    parser = argparse.ArgumentParser(
        description="Run the single pretraining flow directly from raw SpatialVID videos and captions."
    )
    parser.add_argument("--config", required=True)
    parser.add_argument("--model", help="Shared model profile YAML")
    parser.add_argument("--log-level", choices=LOG_LEVELS, default="INFO")
    parser.add_argument("--set", action="append", default=[])
    parser.add_argument("--gpus", type=int)
    parser.add_argument("--nodes", type=int, default=1)
    parser.add_argument("--node-rank", type=int, default=0)
    parser.add_argument("--master-addr", default="127.0.0.1")
    parser.add_argument("--master-port", type=int, default=29500)
    parser.add_argument("--worker", action="store_true", help=argparse.SUPPRESS)
    args = parser.parse_args(argv)
    from alc_video.runtime.cli import initialize_cli

    initialize_cli(args.log_level)
    config = Path(args.config).expanduser().resolve()
    from alc_video.training.config import load_run_config

    model_options = (
        {} if args.model is None else {"model": Path(args.model).expanduser().resolve()}
    )
    from alc_video.utils.errors import input_errors

    with input_errors():
        load_run_config(config, tuple(args.set), **model_options)
    LOG.info("launcher config=%s worker=%s", config, args.worker)
    if not args.worker:
        import torch

        gpus = torch.cuda.device_count() if args.gpus is None else args.gpus
        if not 0 < gpus <= torch.cuda.device_count():
            parser.error("--gpus must select available CUDA devices")
        if args.nodes <= 0 or not 0 <= args.node_rank < args.nodes:
            parser.error("--nodes and --node-rank must identify a valid topology")
        command = [
            sys.executable,
            "-m",
            "accelerate.commands.launch",
            "--use_deepspeed",
            "--gpu_ids",
            "all",
            "--num_processes",
            str(gpus * args.nodes),
            "--num_machines",
            str(args.nodes),
            "--main_process_port",
            str(args.master_port),
            "--mixed_precision",
            "bf16",
            "--dynamo_backend",
            "no",
        ]
        if args.nodes > 1:
            command.extend(
                [
                    "--deepspeed_multinode_launcher",
                    "standard",
                    "--same_network",
                    "--machine_rank",
                    str(args.node_rank),
                    "--main_process_ip",
                    args.master_addr,
                ]
            )
        command.extend(
            [
                "--module",
                "scripts.train",
                "--worker",
                "--config",
                str(config),
                "--log-level",
                args.log_level,
            ]
        )
        if args.model is not None:
            command.extend(["--model", str(model_options["model"])])
        for override in args.set:
            command.extend(["--set", override])
        environment = dict(os.environ)

        environment["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"
        environment["NCCL_ALGO"] = "Ring"
        environment["NCCL_PROTO"] = "Simple"
        environment.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")
        os.execvpe(command[0], command, environment)
    start_method = multiprocessing.get_start_method(allow_none=True)
    if start_method is None:
        multiprocessing.set_start_method("spawn")
    elif start_method != "spawn":
        raise RuntimeError("DataLoader multiprocessing start method must be spawn")
    from alc_video.training.execution.loop import train_from_config

    result = train_from_config(
        config, tuple(args.set), log_level=args.log_level, **model_options
    )
    LOG.debug("worker returned checkpoint=%s", result)
