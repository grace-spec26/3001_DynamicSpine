"""Run ShapeWorks with a controlled retry or stage a validated particle fallback."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
import shutil
import subprocess
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable

import numpy as np
from openpyxl import load_workbook

from completion_inventory import LEVELS
from common import output_relative_path
from pipeline_provenance import validate_materialized, write_manifest


def _expected_paths(
    root: Path, case_ids: list[str], domains: list[str]
) -> list[Path]:
    return [
        Path(root) / f"{case_id}_{domain}_{coordinate_system}.particles"
        for case_id in case_ids
        for domain in domains
        for coordinate_system in ("local", "world")
    ]


def validate_particle_set(
    root: Path,
    case_ids: list[str],
    domains: list[str],
    particles_per_domain: int = 256,
    allow_unrelated: bool = False,
) -> dict[str, Any]:
    """Validate an exact case-major/domain-major local/world particle set."""
    root = Path(root)
    expected = _expected_paths(root, case_ids, domains)
    missing = [path for path in expected if not path.is_file()]
    if missing:
        raise ValueError("missing particle files: " + ", ".join(map(str, missing)))
    if not allow_unrelated:
        actual = set(root.glob("*.particles"))
        extra = sorted(actual.difference(expected))
        if extra:
            raise ValueError("unexpected particle files: " + ", ".join(map(str, extra)))

    for path in expected:
        validate_materialized(path)
        try:
            values = np.loadtxt(path, dtype=float)
        except (OSError, ValueError) as exc:
            raise ValueError(f"malformed particle file {path}: {exc}") from exc
        values = np.atleast_2d(values)
        if values.shape != (particles_per_domain, 3):
            raise ValueError(
                f"{path}: expected ({particles_per_domain}, 3), got {values.shape}"
            )
        if not np.isfinite(values).all():
            raise ValueError(f"{path}: particles must be finite")
    return {
        "case_ids": list(case_ids),
        "domains": list(domains),
        "particles_per_domain": particles_per_domain,
        "file_count": len(expected),
        "paths": [str(path) for path in expected],
    }


def validate_particle_order(
    order_path: Path, case_ids: list[str], domains: list[str]
) -> dict[str, Any]:
    """Require independently recorded reference case/domain order to match."""
    payload = json.loads(Path(order_path).read_text(encoding="utf-8"))
    if payload.get("case_ids") != list(case_ids):
        raise ValueError("reference particle case order mismatch")
    if payload.get("domains") != list(domains):
        raise ValueError("reference particle domain order mismatch")
    return payload


def validate_particle_store(
    store_path: Path,
    case_ids: list[str],
    domains: list[str],
    particles_per_domain: int = 256,
) -> dict[str, Any]:
    """Fully validate the durable NPZ, including exact semantic ordering."""
    with np.load(Path(store_path), allow_pickle=False) as bundle:
        stored_cases = bundle["case_ids"].astype(str).tolist()
        stored_domains = bundle["domains"].astype(str).tolist()
        stored_count = int(np.asarray(bundle["particles_per_domain"]).item())
        local = np.asarray(bundle["local"], dtype=float)
        world = np.asarray(bundle["world"], dtype=float)
    if stored_cases != list(case_ids):
        raise ValueError("particle store case order mismatch")
    if stored_domains != list(domains):
        raise ValueError("particle store domain order mismatch")
    if stored_count != particles_per_domain:
        raise ValueError("particle store row-count mismatch")
    expected = (len(case_ids), len(domains), particles_per_domain, 3)
    if local.shape != expected or world.shape != expected:
        raise ValueError(f"particle store arrays must have shape {expected}")
    if not np.isfinite(local).all() or not np.isfinite(world).all():
        raise ValueError("particle store values must be finite")
    return {
        "case_ids": stored_cases,
        "domains": stored_domains,
        "particles_per_domain": stored_count,
        "local": local,
        "world": world,
    }


def build_particle_store(
    particle_root: Path,
    store_path: Path,
    case_ids: list[str],
    domains: list[str],
    particles_per_domain: int = 256,
) -> Path:
    """Publish one durable ordered NPZ as the analysis particle source."""
    validate_particle_set(
        particle_root, case_ids, domains, particles_per_domain=particles_per_domain
    )
    arrays: dict[str, np.ndarray] = {}
    for coordinate_system in ("local", "world"):
        arrays[coordinate_system] = np.stack(
            [
                np.loadtxt(
                    Path(particle_root)
                    / f"{case_id}_{domain}_{coordinate_system}.particles",
                    dtype=float,
                )
                for case_id in case_ids
                for domain in domains
            ]
        ).reshape(len(case_ids), len(domains), particles_per_domain, 3)
    store_path = Path(store_path)
    store_path.parent.mkdir(parents=True, exist_ok=True)
    partial = store_path.with_name(store_path.name + ".partial.npz")
    np.savez_compressed(
        partial,
        case_ids=np.asarray(case_ids),
        domains=np.asarray(domains),
        particles_per_domain=np.asarray(particles_per_domain),
        local=arrays["local"],
        world=arrays["world"],
    )
    validate_particle_store(
        partial, case_ids, domains, particles_per_domain=particles_per_domain
    )
    os.replace(partial, store_path)
    return store_path


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _atomic_copy(source: Path, destination: Path) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    partial = destination.with_name(f"{destination.name}.partial")
    try:
        shutil.copy2(source, partial)
        if _sha256(source) != _sha256(partial):
            raise ValueError(f"copied particle differs from source: {source}")
        os.replace(partial, destination)
    finally:
        if partial.exists():
            partial.unlink()


def stage_reference_particles(
    reference: Path,
    destination: Path,
    case_ids: list[str],
    domains: list[str],
    particles_per_domain: int = 256,
    order_path: Path | None = None,
) -> list[Path]:
    """Validate and atomically copy only the requested reference cohort."""
    reference = Path(reference)
    destination = Path(destination)
    if order_path is not None:
        validate_particle_order(order_path, case_ids, domains)
    metadata = validate_particle_set(
        reference, case_ids, domains, particles_per_domain=particles_per_domain
    )
    copied: list[Path] = []
    for source_value in metadata["paths"]:
        source = Path(source_value)
        target = destination / source.name
        _atomic_copy(source, target)
        copied.append(target)
    validate_particle_set(
        destination,
        case_ids,
        domains,
        particles_per_domain=particles_per_domain,
        allow_unrelated=True,
    )
    return copied


def write_particle_validation_cache(
    source_root: Path,
    cache_path: Path,
    case_ids: list[str],
    domains: list[str],
    particles_per_domain: int = 256,
) -> Path:
    """Cache exact names, logical sizes, and hashes after full validation."""
    source_root = Path(source_root)
    metadata = validate_particle_set(
        source_root,
        case_ids,
        domains,
        particles_per_domain=particles_per_domain,
    )
    files = []
    for value in metadata["paths"]:
        path = Path(value)
        files.append(
            {"name": path.name, "size": path.stat().st_size, "sha256": _sha256(path)}
        )
    payload = {
        "case_ids": list(case_ids),
        "domains": list(domains),
        "particles_per_domain": particles_per_domain,
        "files": files,
    }
    cache_path = Path(cache_path)
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    cache_path.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    return cache_path


def validate_particle_cache(
    published_root: Path,
    cache_path: Path,
    case_ids: list[str],
    domains: list[str],
    particles_per_domain: int = 256,
    content_root: Path | None = None,
    verify_content: bool = True,
) -> dict[str, Any]:
    """Validate mirror inventory; optionally re-hash a materialized content root."""
    published_root = Path(published_root)
    cache = json.loads(Path(cache_path).read_text(encoding="utf-8"))
    if cache.get("case_ids") != list(case_ids):
        raise ValueError("particle cache case order mismatch")
    if cache.get("domains") != list(domains):
        raise ValueError("particle cache domain order mismatch")
    if cache.get("particles_per_domain") != particles_per_domain:
        raise ValueError("particle cache row-count mismatch")
    expected_names = [path.name for path in _expected_paths(published_root, case_ids, domains)]
    records = cache.get("files", [])
    if [record.get("name") for record in records] != expected_names:
        raise ValueError("particle cache filename order mismatch")
    actual_names = sorted(path.name for path in published_root.glob("*.particles"))
    if actual_names != sorted(expected_names):
        raise ValueError("published particle filename inventory mismatch")
    content_root = Path(content_root) if content_root is not None else published_root
    for record in records:
        published = published_root / record["name"]
        if published.stat().st_size != record["size"]:
            raise ValueError(f"particle size mismatch: {published}")
        if not verify_content:
            continue
        content = content_root / record["name"]
        if not content.is_file() or content.stat().st_size != record["size"]:
            raise ValueError(f"validated content size mismatch: {content}")
        if _sha256(content) != record["sha256"]:
            raise ValueError(f"particle checksum mismatch: {content}")
    return {
        "case_ids": list(case_ids),
        "domains": list(domains),
        "particles_per_domain": particles_per_domain,
        "file_count": len(records),
        "content_root": str(content_root),
    }


def run_shapeworks(
    project: Path, executable: Path, log: Path
) -> subprocess.CompletedProcess[str]:
    """Run one ShapeWorks optimization attempt and persist combined output."""
    project = Path(project)
    executable = Path(executable)
    log = Path(log)
    validate_materialized(project)
    if not executable.is_file() or not os.access(executable, os.X_OK):
        raise ValueError(f"ShapeWorks executable is not runnable: {executable}")
    result = subprocess.run(
        [str(executable), "optimize", "--name", str(project)],
        check=False,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
    )
    log.parent.mkdir(parents=True, exist_ok=True)
    log.write_text(result.stdout or "", encoding="utf-8")
    return result


@dataclass(frozen=True)
class AttemptResult:
    args: list[str]
    returncode: int
    stdout: str
    valid: bool
    error: str | None = None


def run_shapeworks_with_retry(
    project: Path,
    executable: Path,
    logs_dir: Path,
    attempts: int = 2,
    validate_attempt: Callable[[], None] | None = None,
    reset_attempt: Callable[[int], None] | None = None,
) -> list[AttemptResult]:
    """Retry launch/read/validation failures and accept only validated output."""
    if attempts < 1:
        raise ValueError("attempts must be at least 1")
    logs_dir = Path(logs_dir)
    results: list[AttemptResult] = []
    for attempt in range(1, attempts + 1):
        log = logs_dir / f"shapeworks_attempt_{attempt}.log"
        try:
            if reset_attempt is not None:
                reset_attempt(attempt)
            process = run_shapeworks(project, executable, log)
            valid = process.returncode == 0
            error = None
            if valid and validate_attempt is not None:
                try:
                    validate_attempt()
                except Exception as exc:
                    valid = False
                    error = str(exc)
                    with log.open("a", encoding="utf-8") as handle:
                        handle.write(f"\nArtifact validation failed: {exc}\n")
            result = AttemptResult(
                list(map(str, process.args)),
                process.returncode,
                process.stdout or "",
                valid,
                error,
            )
        except Exception as exc:
            log.parent.mkdir(parents=True, exist_ok=True)
            log.write_text(f"ShapeWorks attempt failed: {exc}\n", encoding="utf-8")
            result = AttemptResult(
                [str(executable), "optimize", "--name", str(project)],
                1,
                "",
                False,
                str(exc),
            )
        results.append(result)
        if result.valid:
            break
    return results


def _case_ids(workshop: Path) -> list[str]:
    with (workshop / "data" / "cases.csv").open(
        newline="", encoding="utf-8"
    ) as handle:
        return [row["case_id"] for row in csv.DictReader(handle)]


def _load_manifest(path: Path) -> dict[str, Any]:
    if path.is_file():
        return json.loads(path.read_text(encoding="utf-8"))
    return {"commands": [], "versions": {}, "failures": [], "artifacts": {}}


def build_reference_order_metadata(reference_root: Path, output_path: Path) -> Path:
    """Record order from the reference cohort and workbook/template headers."""
    reference_root = Path(reference_root)
    candidates = (
        reference_root / "outputs" / "full_run.xlsx",
        reference_root / "outputs" / "shapeworks_all.xlsx",
    )
    workbook_path = next((path for path in candidates if path.is_file()), None)
    if workbook_path is None:
        raise ValueError("reference order cannot be verified without a reference workbook")
    sheet = load_workbook(workbook_path, read_only=True, data_only=True)["data"]
    rows = list(sheet.iter_rows(min_row=2, values_only=True))
    case_ids = [str(row[0]) for row in rows]
    if not case_ids or len(set(case_ids)) != len(case_ids):
        raise ValueError("reference workbook case order is missing or duplicated")
    cohort_path = reference_root / "data" / "cases.csv"
    if cohort_path.is_file():
        with cohort_path.open(newline="", encoding="utf-8") as handle:
            cohort_ids = [row["case_id"] for row in csv.DictReader(handle)]
        if cohort_ids != case_ids:
            raise ValueError("reference workbook and cohort case order disagree")
    first_paths = [
        output_relative_path(value, workbook_path.parent)
        for value in rows[0][2:17]
        if value not in (None, "")
    ]
    domains = [path.stem.rsplit("_", 1)[-1] for path in first_paths]
    if domains != list(LEVELS):
        raise ValueError(f"reference workbook domain order is invalid: {domains}")
    payload = {
        "case_ids": case_ids,
        "domains": domains,
        "source_workbook": workbook_path.relative_to(reference_root).as_posix(),
    }
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    return output_path


def _reference_commit(reference_root: Path) -> str | None:
    result = subprocess.run(
        ["git", "-C", str(reference_root), "rev-parse", "HEAD"],
        check=False,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.DEVNULL,
    )
    return result.stdout.strip() if result.returncode == 0 else None


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--workshop", type=Path, required=True)
    parser.add_argument("--reference", type=Path, required=True)
    parser.add_argument(
        "--executable",
        type=Path,
        default=Path("/Applications/ShapeWorks/bin/shapeworks"),
    )
    parser.add_argument("--allow-reference-fallback", action="store_true")
    args = parser.parse_args(argv)

    workshop = args.workshop.resolve()
    output = workshop / "outputs"
    destination = output / "full_run_particles"
    case_ids = _case_ids(workshop)
    project = output / "full_run.xlsx"
    if not project.is_file():
        project = output / "shapeworks_all.xlsx"
    quarantine = output / "logs" / "particle_attempts"

    def reset_attempt(attempt: int) -> None:
        if destination.exists():
            quarantine.mkdir(parents=True, exist_ok=True)
            target = quarantine / f"attempt_{attempt}_preexisting"
            if target.exists():
                shutil.rmtree(target)
            shutil.move(str(destination), target)

    results = run_shapeworks_with_retry(
        project,
        args.executable,
        output / "logs",
        attempts=2,
        validate_attempt=lambda: validate_particle_set(
            destination, case_ids, list(LEVELS)
        ),
        reset_attempt=reset_attempt,
    )
    manifest_path = output / "provenance_manifest.json"
    manifest = _load_manifest(manifest_path)
    manifest.setdefault("commands", []).extend(
        " ".join(map(str, result.args)) for result in results
    )
    if results[-1].valid:
        validation = validate_particle_set(destination, case_ids, list(LEVELS))
        source = "workshop_regenerated"
    elif args.allow_reference_fallback:
        if destination.exists():
            quarantine.mkdir(parents=True, exist_ok=True)
            target = quarantine / "failed_final_output"
            if target.exists():
                shutil.rmtree(target)
            shutil.move(str(destination), target)
        reference_particles = args.reference / "outputs" / "full_run_particles"
        if not reference_particles.is_dir():
            reference_particles = args.reference
        order_path = build_reference_order_metadata(
            args.reference, output / "reference_particle_order.json"
        )
        validate_particle_order(order_path, case_ids, list(LEVELS))
        copied = stage_reference_particles(
            reference_particles,
            destination,
            case_ids,
            list(LEVELS),
            order_path=order_path,
        )
        validation = validate_particle_set(destination, case_ids, list(LEVELS))
        source = "reference_fallback"
        manifest.setdefault("failures", []).append(
            "ShapeWorks optimization failed after two attempts; staged validated reference particles"
        )
        manifest.setdefault("fallback", {})["particle_source"] = {
            "path": str(reference_particles),
            "file_count": len(copied),
            "order_metadata": str(order_path),
        }
    else:
        return results[-1].returncode or 1

    store = build_particle_store(
        destination,
        output / "particle_store.npz",
        case_ids,
        list(LEVELS),
    )
    write_particle_validation_cache(
        destination,
        output / "full_run_particles.validation.json",
        case_ids,
        list(LEVELS),
    )
    artifact = {
        "source": source,
        "file_count": validation["file_count"],
        "particles_per_domain": validation["particles_per_domain"],
        "case_count": len(case_ids),
        "domain_count": len(LEVELS),
        "analysis_source": str(store),
    }
    if source == "reference_fallback":
        artifact["reference_path"] = str(reference_particles)
        commit = _reference_commit(args.reference)
        if commit:
            artifact["reference_commit"] = commit
    manifest.setdefault("artifacts", {})["full_run_particles"] = artifact
    write_manifest(manifest_path, manifest)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
