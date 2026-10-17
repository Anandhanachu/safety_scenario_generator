"""Explicit installed-file registration; never treats file existence as inference readiness."""
from pathlib import Path

from backend.configuration import DATA_DIR, defaults
from backend.persistence import Store


def store():
    return Store(DATA_DIR / "agent-models.sqlite")


def register_file(path):
    candidate = Path(path).expanduser().resolve()
    roots = [Path(p).expanduser().resolve() for p in defaults()["model_dirs"]]
    if not candidate.is_file() or candidate.suffix.lower() != ".gguf" or not any(candidate.is_relative_to(r) for r in roots):
        raise ValueError("Model must be an existing GGUF inside a permitted board model directory")
    registry = store()
    paths = registry.setting("registered_files", [])
    if str(candidate) not in paths:
        registry.set_setting("registered_files", paths + [str(candidate)])
    return {"id": str(candidate), "path": str(candidate), "size_bytes": candidate.stat().st_size,
            "format": "GGUF", "status": "registered_file_runtime_unverified", "inference_ready": False}


def registered_files():
    results = []
    for path in store().setting("registered_files", [])[:200]:
        p = Path(path)
        results.append({"id": path, "path": path, "format": "GGUF", "size_bytes": p.stat().st_size if p.is_file() else None,
                        "status": "registered_file_runtime_unverified" if p.is_file() else "registered_file_missing", "inference_ready": False})
    return results
