from dataclasses import asdict
from pathlib import Path
from urllib.parse import quote

from alc_video.conditioning.config import CacheConstructionSettings
from alc_video.inference.examples import discover_cases


def discover_examples(directory: Path) -> list[dict]:
    examples = []
    for case in discover_cases(directory):
        if case.task == "rollout":
            continue
        identifier = case.path.parent.relative_to(directory.resolve()).as_posix()
        media = case.source.path.relative_to(directory.resolve()).as_posix()
        prepare = {
            name: value
            for name, value in asdict(case.source).items()
            if name not in ("path", "type") and value is not None
        }
        examples.append(
            {
                "id": identifier,
                "name": case.path.parent.name.replace("_", " ").replace("-", " "),
                "kind": case.source.type,
                "url": "api/example-files/" + quote(media),
                "prompt": case.prompts[0],
                "profile": case.model.stem,
                "mode": case.mode,
                "prepare": prepare,
                "cache": asdict(case.cache or CacheConstructionSettings()),
                "camera": asdict(case.camera),
                "time": asdict(case.time),
            }
        )
    return examples
