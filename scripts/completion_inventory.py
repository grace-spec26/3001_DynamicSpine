"""Inventory and integrity checks for the complete 71-case course cohort."""

from __future__ import annotations

import csv
import hashlib
import json
from collections import Counter
from pathlib import Path
from typing import Any

import numpy as np

from pipeline_provenance import validate_materialized


LEVELS = tuple([f"T{i}" for i in range(1, 13)] + ["L1", "L2", "L3"])


def _rows(package_root: Path) -> list[dict[str, str]]:
    cases_path = Path(package_root) / "data" / "cases.csv"
    validate_materialized(cases_path)
    with cases_path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def build_inventory(package_root: Path) -> dict[str, Any]:
    """Inspect the expected raw cohort without writing into ``data/``."""
    package_root = Path(package_root)
    rows = _rows(package_root)
    missing: list[str] = []
    invalid: list[str] = []
    mesh_paths: list[Path] = []

    case_ids = [row.get("case_id", "") for row in rows]
    repeated_cases = sorted(
        case_id for case_id, count in Counter(case_ids).items() if count > 1
    )
    if any(not case_id for case_id in case_ids):
        invalid.append("cases.csv contains an empty case_id")
    if repeated_cases:
        invalid.append(f"duplicate case IDs: {', '.join(repeated_cases)}")

    for row in rows:
        case_id = row.get("case_id", "")
        for field in ("mask", "transform", "merged_mesh"):
            value = row.get(field, "")
            path = package_root / value
            if not path.exists():
                missing.append(str(path))
                continue
            try:
                validate_materialized(path)
            except ValueError as exc:
                invalid.append(str(exc))
        transform_value = row.get("transform", "")
        transform_path = package_root / transform_value
        if transform_path.is_file() and transform_path.stat().st_size:
            try:
                payload = json.loads(transform_path.read_text(encoding="utf-8"))
                transform = np.asarray(payload["world_ras_to_aligned"], dtype=float)
                if transform.shape != (4, 4) or not np.isfinite(transform).all():
                    raise ValueError("expected a finite 4x4 world_ras_to_aligned matrix")
            except (OSError, KeyError, TypeError, ValueError, json.JSONDecodeError) as exc:
                invalid.append(f"invalid transform {transform_path}: {exc}")

        for level in LEVELS:
            path = package_root / "data" / "meshes" / case_id / f"{level}.ply"
            mesh_paths.append(path)
            if not path.exists():
                missing.append(str(path))
                continue
            try:
                validate_materialized(path)
            except ValueError as exc:
                invalid.append(str(exc))

    basename_counts = Counter(path.name for path in mesh_paths if path.exists())
    return {
        "case_ids": case_ids,
        "domains": list(LEVELS),
        "mesh_count": sum(path.is_file() and path.stat().st_size > 0 for path in mesh_paths),
        "duplicate_basenames": sorted(
            name for name, count in basename_counts.items() if count > 1
        ),
        "missing": missing,
        "invalid": invalid,
    }


def validate_cohort(
    inventory: dict[str, Any], expected_cases: int = 71, expected_domains: int = 15
) -> None:
    """Validate cohort cardinality and every required materialized input."""
    errors: list[str] = []
    if len(inventory.get("case_ids", [])) != expected_cases:
        errors.append(
            f"expected {expected_cases} cases, got {len(inventory.get('case_ids', []))}"
        )
    if len(inventory.get("domains", [])) != expected_domains:
        errors.append(
            f"expected {expected_domains} domains, got {len(inventory.get('domains', []))}"
        )
    expected_meshes = expected_cases * expected_domains
    if inventory.get("mesh_count") != expected_meshes:
        errors.append(
            f"expected {expected_meshes} valid meshes, got {inventory.get('mesh_count')}"
        )
    if inventory.get("missing"):
        errors.append("missing inputs: " + ", ".join(inventory["missing"]))
    if inventory.get("invalid"):
        errors.append("invalid inputs: " + ", ".join(inventory["invalid"]))
    if errors:
        raise ValueError("; ".join(errors))


def verify_sha256_manifest(package_root: Path, checksum_file: Path) -> None:
    """Verify a standard ``SHA256SUMS.txt`` against *package_root*."""
    package_root = Path(package_root)
    checksum_file = Path(checksum_file)
    validate_materialized(checksum_file)
    for line_number, line in enumerate(
        checksum_file.read_text(encoding="utf-8").splitlines(), start=1
    ):
        if not line.strip():
            continue
        try:
            expected, relative = line.split("  ", 1)
        except ValueError as exc:
            raise ValueError(f"invalid checksum line {line_number}") from exc
        path = package_root / relative
        validate_materialized(path)
        digest = hashlib.sha256()
        with path.open("rb") as handle:
            for chunk in iter(lambda: handle.read(1024 * 1024), b""):
                digest.update(chunk)
        if digest.hexdigest() != expected:
            raise ValueError(f"checksum mismatch: {relative}")
