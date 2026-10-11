from pathlib import Path
from urllib.parse import quote

IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png", ".webp"}
VIDEO_SUFFIXES = {".mp4", ".webm", ".mov", ".mkv"}


def discover_examples(directory: Path) -> list[dict]:
    examples = []
    for path in sorted(directory.rglob("*")):
        suffix = path.suffix.lower()
        if not path.is_file() or suffix not in IMAGE_SUFFIXES | VIDEO_SUFFIXES:
            continue
        name = path.relative_to(directory).as_posix()
        prompt = path.with_suffix(".txt")
        examples.append(
            {
                "id": name,
                "name": path.stem.replace("_", " ").replace("-", " "),
                "kind": "image" if suffix in IMAGE_SUFFIXES else "video",
                "url": "api/example-files/" + quote(name),
                "prompt": prompt.read_text().strip() if prompt.is_file() else "",
            }
        )
    return examples
