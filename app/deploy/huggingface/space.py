"""Hugging Face Space entrypoint; source code remains in the GitHub repository."""

import subprocess
import sys
from pathlib import Path

REPOSITORY = "https://github.com/ZGCA-HMI-Lab/4LC.git"
SOURCE = Path(__file__).resolve().parent / "source"

if SOURCE.exists():
    subprocess.run(
        ["git", "-C", str(SOURCE), "pull", "--ff-only", "origin", "main"],
        check=True,
    )
else:
    subprocess.run(
        ["git", "clone", "--depth", "1", "--branch", "main", REPOSITORY, str(SOURCE)],
        check=True,
    )

subprocess.run(["npm", "ci", "--prefix", str(SOURCE / "app")], check=True)
subprocess.run(
    ["npm", "run", "build", "--prefix", str(SOURCE / "app")], check=True
)
sys.path.insert(0, str(SOURCE / "app"))
from deploy.huggingface.assets import prepare_assets

prepare_assets(SOURCE)
sys.argv = [
    "4lc-space",
    "--core-root",
    str(SOURCE),
    "--host",
    "0.0.0.0",
    "--port",
    "7860",
    "--zerogpu",
]
from server.__main__ import main

main()
