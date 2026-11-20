"""Validation and provenance helpers for the completion pipeline."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any


VALID_SOURCES = {
    "workshop_regenerated",
    "workshop_reused",
    "reference_fallback",
}


def validate_materialized(path: Path) -> None:
    """Raise when *path* is not a readable, non-empty regular file."""
    path = Path(path)
    if not path.is_file():
        raise ValueError(f"not a materialized file: {path}")
    try:
        if path.stat().st_size <= 0:
            raise ValueError(f"not a materialized file: {path}")
        with path.open("rb") as handle:
            handle.read(1)
    except OSError as exc:
        raise ValueError(f"not a materialized file: {path}: {exc}") from exc


def inspect_file(path: Path) -> dict[str, Any]:
    """Return stable metadata after fully reading a valid artifact."""
    path = Path(path)
    validate_materialized(path)
    digest = hashlib.sha256()
    try:
        with path.open("rb") as handle:
            for chunk in iter(lambda: handle.read(1024 * 1024), b""):
                digest.update(chunk)
    except OSError as exc:
        raise ValueError(f"not a materialized file: {path}: {exc}") from exc
    return {
        "path": str(path),
        "size": path.stat().st_size,
        "sha256": digest.hexdigest(),
    }


def record_artifact(
    manifest: dict[str, Any],
    family: str,
    source: str,
    paths: list[Path],
    metadata: dict[str, Any],
) -> None:
    """Record one validated artifact family in *manifest*."""
    if source not in VALID_SOURCES:
        raise ValueError(f"invalid artifact source: {source}")
    if not family:
        raise ValueError("artifact family must not be empty")
    files = [inspect_file(Path(path)) for path in paths]
    manifest.setdefault("artifacts", {})[family] = {
        "source": source,
        "files": files,
        **metadata,
    }


def write_manifest(path: Path, manifest: dict[str, Any]) -> None:
    """Write a deterministic JSON manifest, creating its parent directory."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
