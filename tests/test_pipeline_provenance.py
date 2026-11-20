import hashlib
import json
from pathlib import Path

import pytest


from pipeline_provenance import (
    inspect_file,
    record_artifact,
    validate_materialized,
    write_manifest,
)


def test_inspect_file_reports_size_and_content_hash(tmp_path: Path) -> None:
    artifact = tmp_path / "artifact.txt"
    artifact.write_bytes(b"spine-data")

    result = inspect_file(artifact)

    assert result["path"] == str(artifact)
    assert result["size"] == 10
    assert result["sha256"] == hashlib.sha256(b"spine-data").hexdigest()


@pytest.mark.parametrize("kind", ["missing", "empty", "directory"])
def test_validate_materialized_rejects_unusable_paths(
    tmp_path: Path, kind: str
) -> None:
    candidate = tmp_path / "candidate"
    if kind == "empty":
        candidate.touch()
    elif kind == "directory":
        candidate.mkdir()

    with pytest.raises(ValueError, match="materialized file"):
        validate_materialized(candidate)


def test_record_artifact_rejects_unknown_source(tmp_path: Path) -> None:
    artifact = tmp_path / "particle.particles"
    artifact.write_text("0 0 0\n", encoding="utf-8")

    with pytest.raises(ValueError, match="source"):
        record_artifact({}, "particles", "copied_somewhere", [artifact], {})


def test_manifest_round_trip_keeps_provenance_fields(tmp_path: Path) -> None:
    artifact = tmp_path / "particle.particles"
    artifact.write_text("0 0 0\n", encoding="utf-8")
    manifest = {
        "commands": ["shapeworks optimize"],
        "versions": {"python": "3.10"},
        "failures": ["first ShapeWorks attempt exited 1"],
    }
    record_artifact(
        manifest,
        "particles",
        "reference_fallback",
        [artifact],
        {"reason": "controlled retry failed", "timestamp": "2026-11-19T12:00:00Z"},
    )
    destination = tmp_path / "provenance_manifest.json"

    write_manifest(destination, manifest)

    loaded = json.loads(destination.read_text(encoding="utf-8"))
    assert loaded["commands"] == ["shapeworks optimize"]
    assert loaded["versions"] == {"python": "3.10"}
    assert loaded["failures"] == ["first ShapeWorks attempt exited 1"]
    assert loaded["artifacts"]["particles"]["source"] == "reference_fallback"
    assert loaded["artifacts"]["particles"]["files"][0]["sha256"] == hashlib.sha256(
        b"0 0 0\n"
    ).hexdigest()
