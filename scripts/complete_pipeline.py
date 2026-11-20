"""Resumable orchestration and final validation for the completed workshop."""

from __future__ import annotations

import csv
import json
import subprocess
import sys
from datetime import datetime, timezone
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable

import numpy as np
from openpyxl import load_workbook

from common import LEVELS
from model_validation import load_validated_model_bundle
from shapeworks_completion import validate_particle_cache, validate_particle_store


@dataclass(frozen=True)
class PipelineContext:
    workshop: Path
    reference: Path
    allow_reference_fallback: bool = False


@dataclass(frozen=True)
class Stage:
    name: str
    action: Callable[[PipelineContext], None]
    validate: Callable[[PipelineContext], None]


def _write_state(path: Path, state: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(state, indent=2) + "\n", encoding="utf-8")


def _record_stage(context: PipelineContext, stage: str) -> None:
    path = context.workshop / "outputs" / "provenance_manifest.json"
    manifest = json.loads(path.read_text(encoding="utf-8")) if path.is_file() else {}
    artifact = manifest.setdefault("artifacts", {}).setdefault(stage, {})
    artifact.update(
        source="workshop_regenerated",
        completed_at=datetime.now(timezone.utc).isoformat(),
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def run_stages(
    stages: list[Stage],
    context: PipelineContext,
    state_path: Path,
    resume: bool = False,
) -> dict[str, Any]:
    """Run stages in order, validating every skipped or completed stage."""
    state_path = Path(state_path)
    if state_path.is_file():
        state = json.loads(state_path.read_text(encoding="utf-8"))
    else:
        state = {"completed": [], "skipped": [], "failed": None}
    state.setdefault("completed", [])
    state["skipped"] = []
    state["failed"] = None

    upstream_rebuilt = False
    for stage in stages:
        if resume and not upstream_rebuilt:
            try:
                stage.validate(context)
            except (OSError, ValueError, KeyError, AssertionError):
                pass
            else:
                if stage.name not in state["completed"]:
                    state["completed"].append(stage.name)
                state["skipped"].append(stage.name)
                _write_state(state_path, state)
                continue
        try:
            stage.action(context)
            stage.validate(context)
        except Exception as exc:
            state["failed"] = {"stage": stage.name, "error": str(exc)}
            _write_state(state_path, state)
            raise
        if stage.name not in state["completed"]:
            state["completed"].append(stage.name)
        upstream_rebuilt = True
        _record_stage(context, stage.name)
        _write_state(state_path, state)
    return state


def summarize_sources(manifest: dict[str, Any]) -> dict[str, int]:
    """Count artifact families by their top-level provenance source."""
    counts = Counter(
        artifact.get("source", "unknown")
        for artifact in manifest.get("artifacts", {}).values()
    )
    return dict(sorted(counts.items()))


def _case_ids(workshop: Path) -> list[str]:
    with (workshop / "data" / "cases.csv").open(
        newline="", encoding="utf-8"
    ) as handle:
        return [row["case_id"] for row in csv.DictReader(handle)]


def _run(context: PipelineContext, stage: str, *arguments: str) -> None:
    log = context.workshop / "outputs" / "logs" / f"{stage}.log"
    log.parent.mkdir(parents=True, exist_ok=True)
    result = subprocess.run(
        [sys.executable, *arguments],
        cwd=context.workshop,
        check=False,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
    )
    log.write_text(result.stdout or "", encoding="utf-8")
    if result.returncode:
        raise ValueError(
            f"{stage} exited {result.returncode}; see {log.relative_to(context.workshop)}"
        )


def _validate_exercises(context: PipelineContext) -> None:
    _run(context, "exercises", "scripts/check_exercises.py")


def _validate_projects(context: PipelineContext) -> None:
    output = context.workshop / "outputs"
    centroids = list((output / "centroids").glob("*.json"))
    if len(centroids) != 71:
        raise ValueError(f"expected 71 centroids, got {len(centroids)}")
    for path in centroids:
        points = np.asarray(json.loads(path.read_text(encoding="utf-8"))["aligned_mm"])
        if points.shape != (15, 3) or not np.isfinite(points).all():
            raise ValueError(f"invalid centroid file: {path}")
    expected_rows = {
        "shapeworks_all.xlsx": 71,
        "shapeworks_demo.xlsx": 8,
        "full_run.xlsx": 71,
    }
    expected_ids = _case_ids(context.workshop)
    for name, row_count in expected_rows.items():
        path = output / name
        if not path.is_file() or path.stat().st_size == 0:
            raise ValueError(f"missing project workbook: {path}")
        sheet = load_workbook(path, read_only=True, data_only=True)["data"]
        rows = list(sheet.iter_rows(min_row=2, values_only=True))
        if len(rows) != row_count or any(len(row) < 17 for row in rows):
            raise ValueError(f"invalid project workbook dimensions: {path}")
        if name != "shapeworks_demo.xlsx" and [str(row[0]) for row in rows] != expected_ids:
            raise ValueError(f"project workbook case order mismatch: {path}")
        if any(any(value in (None, "") for value in row[2:17]) for row in rows):
            raise ValueError(f"project workbook has missing domain paths: {path}")


def _action_projects(context: PipelineContext) -> None:
    _run(context, "centroids", "scripts/01_extract_centroids.py")
    _run(context, "project_full", "scripts/02_create_project.py", "--subset", "all")
    _run(context, "project_demo", "scripts/02_create_project.py", "--subset", "demo")
    _run(
        context,
        "project_full_run",
        "scripts/02_create_project.py",
        "--subset",
        "all",
        "--unique-mesh-names",
        "--output-name",
        "full_run.xlsx",
    )


def _validate_particles(context: PipelineContext) -> None:
    output = context.workshop / "outputs"
    case_ids = _case_ids(context.workshop)
    cache = output / "full_run_particles.validation.json"
    validate_particle_store(
        output / "particle_store.npz", case_ids, list(LEVELS)
    )
    validate_particle_cache(
        output / "full_run_particles",
        cache,
        case_ids,
        list(LEVELS),
        verify_content=False,
    )


def _action_particles(context: PipelineContext) -> None:
    arguments = [
        "scripts/shapeworks_completion.py",
        "--workshop",
        str(context.workshop),
        "--reference",
        str(context.reference),
    ]
    if context.allow_reference_fallback:
        arguments.append("--allow-reference-fallback")
    _run(context, "particles", *arguments)


def _validate_pca(context: PipelineContext) -> None:
    output = context.workshop / "outputs"
    with np.load(output / "full_pca.npz", allow_pickle=False) as bundle:
        if bundle["case_ids"].shape != (71,):
            raise ValueError("full PCA must contain 71 case IDs")
        if bundle["mean"].shape != (3840, 3):
            raise ValueError("full PCA mean must contain 3,840 particles")
        if bundle["modes"].shape != (70, 11520):
            raise ValueError("full PCA must contain 70 whole-spine modes")
        if bundle["scores"].shape != (71, 70):
            raise ValueError("full PCA score matrix has the wrong shape")
        if not all(np.isfinite(bundle[name]).all() for name in ("mean", "modes", "scores", "eigenvalues")):
            raise ValueError("full PCA contains non-finite values")
        full = {name: np.asarray(bundle[name]) for name in bundle.files}
    with np.load(output / "pca.npz", allow_pickle=False) as bundle:
        if bundle["modes"].shape != (6, 11520) or bundle["scores"].shape != (71, 6):
            raise ValueError("default PCA bundle has the wrong shape")
        if bundle["case_ids"].astype(str).tolist() != full["case_ids"].astype(str).tolist():
            raise ValueError("default PCA case order mismatch")
        for name in ("mean", "modes", "scores", "eigenvalues"):
            expected = full[name] if name == "mean" else full[name][..., :6] if name == "scores" else full[name][:6]
            if np.asarray(bundle[name]).shape != np.asarray(expected).shape or not np.allclose(bundle[name], expected):
                raise ValueError(f"default PCA {name} does not match full PCA")
    export = output / "pca_exports"
    metadata = json.loads((export / "metadata.json").read_text(encoding="utf-8"))
    if metadata.get("case_ids") != full["case_ids"].astype(str).tolist():
        raise ValueError("PCA export metadata case order mismatch")
    with (export / "scores.csv").open(newline="", encoding="utf-8") as handle:
        score_ids = [row["case_id"] for row in csv.DictReader(handle)]
    if score_ids != metadata["case_ids"]:
        raise ValueError("PCA score export case order mismatch")
    if len(list((export / "modes").glob("pc*.txt"))) != 70:
        raise ValueError("PCA exports must contain all 70 modes")
    eigenvalues = np.atleast_1d(np.loadtxt(export / "eigenvalues.txt"))
    if eigenvalues.shape != (70,) or not np.isfinite(eigenvalues).all():
        raise ValueError("PCA eigenvalue export is incomplete")


def _action_pca(context: PipelineContext) -> None:
    _run(context, "pca_prepare", "scripts/prepare_full_pca_exports.py")
    _run(
        context,
        "pca_default",
        "scripts/03_import_pca_exports.py",
        "--modes",
        "6",
    )
    _run(
        context,
        "pca_full",
        "scripts/03_import_pca_exports.py",
        "--modes",
        "70",
        "--output",
        "outputs/full_pca.npz",
    )


def _validate_regression(context: PipelineContext) -> None:
    output = context.workshop / "outputs" / "regression"
    report = json.loads((output / "report.json").read_text(encoding="utf-8"))
    if [trial["pls_components"] for trial in report["cv"]] != [1, 2, 3, 5, 8]:
        raise ValueError("regression report does not contain the full component grid")
    with np.load(output / "predicted_scores.npz", allow_pickle=False) as prediction:
        if prediction["predicted"].shape != (14, 6):
            raise ValueError("expected six-mode predictions for 14 held-out cases")
        if not np.isfinite(prediction["predicted"]).all():
            raise ValueError("predicted scores contain non-finite values")


def _action_regression(context: PipelineContext) -> None:
    _run(context, "regression", "scripts/04_train_pls.py")


def _validate_reconstructions(context: PipelineContext) -> None:
    folder = context.workshop / "outputs" / "reconstructions"
    with np.load(folder / "reconstructions.npz", allow_pickle=False) as bundle:
        if bundle["predicted"].shape != (14, 3840, 3):
            raise ValueError("expected 14 reconstructed 3,840-particle spines")
        if not np.isfinite(bundle["predicted"]).all():
            raise ValueError("reconstructions contain non-finite values")
        case_ids = bundle["case_ids"].astype(str).tolist()
    for case_id in case_ids:
        if len(list((folder / "particles" / case_id).glob("*.particles"))) != 15:
            raise ValueError(f"incomplete domain reconstruction for {case_id}")


def _action_reconstructions(context: PipelineContext) -> None:
    _run(context, "reconstructions", "scripts/05_reconstruct.py")


def _validate_models(context: PipelineContext) -> None:
    suite = context.workshop / "outputs" / "pca_models"
    for mode_count in (3, 4, 5, 6):
        folder = suite / f"pca_{mode_count:02d}"
        for name in ("pca.npz", "interactive_model.joblib", "reconstructions.npz"):
            path = folder / name
            if not path.is_file() or path.stat().st_size == 0:
                raise ValueError(f"incomplete model suite: missing {path}")
        bundle = load_validated_model_bundle(
            folder / "interactive_model.joblib", folder / "pca.npz"
        )
        if bundle["modes"].shape != (mode_count, 11520):
            raise ValueError(f"invalid PCA model dimensions in {folder}")
    ply = context.workshop / "outputs" / "interactive_particles.ply"
    if not ply.is_file() or "element vertex 3840" not in ply.read_text(
        encoding="ascii"
    ):
        raise ValueError("headless viewer PLY is missing or incomplete")


def _action_models(context: PipelineContext) -> None:
    _run(context, "model_suite", "scripts/build_pca_suite.py")
    _run(
        context,
        "viewer_headless",
        "scripts/07_interactive.py",
        "--run-dir",
        "outputs/pca_models/pca_03",
        "--suite-dir",
        "outputs/pca_models",
        "--headless",
        "--output",
        "outputs/interactive_particles.ply",
    )


def build_stages() -> list[Stage]:
    return [
        Stage("exercises", _validate_exercises, _validate_exercises),
        Stage("projects", _action_projects, _validate_projects),
        Stage("particles", _action_particles, _validate_particles),
        Stage("pca", _action_pca, _validate_pca),
        Stage("regression", _action_regression, _validate_regression),
        Stage("reconstructions", _action_reconstructions, _validate_reconstructions),
        Stage("models", _action_models, _validate_models),
    ]


def validate_completion(workshop: Path) -> None:
    context = PipelineContext(Path(workshop), Path(workshop), False)
    for validator in (
        _validate_projects,
        _validate_particles,
        _validate_pca,
        _validate_regression,
        _validate_reconstructions,
        _validate_models,
    ):
        validator(context)
    manifest = workshop / "outputs" / "provenance_manifest.json"
    if not manifest.is_file():
        raise ValueError("missing provenance manifest")


def completion_exit_code(workshop: Path) -> int:
    try:
        validate_completion(Path(workshop))
    except (OSError, ValueError, KeyError, AssertionError):
        return 1
    return 0


def write_completion_report(
    context: PipelineContext, state: dict[str, Any]
) -> Path:
    output = context.workshop / "outputs"
    manifest = json.loads((output / "provenance_manifest.json").read_text(encoding="utf-8"))
    sources = summarize_sources(manifest)
    report = output / "completion_report.md"
    lines = [
        "# BIOF3001 Workshop Completion Report",
        "",
        "## Result",
        "",
        "The validated end-to-end pipeline is complete for 71 cases and 15 domains.",
        "",
        "## Stage status",
        "",
        f"- Completed: {', '.join(state.get('completed', []))}",
        f"- Resume-skipped after validation: {', '.join(state.get('skipped', [])) or 'none'}",
        "",
        "## Provenance summary",
        "",
    ]
    lines.extend(f"- `{source}`: {count} artifact families" for source, count in sources.items())
    particle = manifest.get("artifacts", {}).get("full_run_particles", {})
    commit = particle.get("reference_commit")
    particle_line = f"Particle source: `{particle.get('source', 'unknown')}`"
    if commit:
        particle_line += f" from reference commit `{commit}`"
    particle_line += f"; analysis source: `{particle.get('analysis_source', 'not recorded')}`."
    failures = list(map(str, manifest.get("failures", [])))
    lines.extend(
        [
            "",
            particle_line,
            "",
            "## Observed failures and fallbacks",
            "",
            *(f"- {failure}" for failure in failures),
            *([] if failures else ["- none"]),
            "",
            "## Verified inventory",
            "",
            "- 71 centroid files",
            "- 2,130 particle files: 71 cases × 15 domains × local/world",
            "- 70 whole-spine PCA modes over 11,520 coordinates",
            "- 57 training and 14 held-out regression cases",
            "- K=3, K=4, K=5, and K=6 interactive model suites",
            "- 3,840-vertex headless PLY export",
            "",
            "## Caveats",
            "",
            f"- Repository status: {manifest.get('repository_status', 'not recorded')}.",
            f"- GUI verification: {manifest.get('gui_verification', 'not recorded')}.",
            "- The fixed 57/14 split evaluates regression within a shared statistical shape model; it is not external validation of the full pipeline.",
        ]
    )
    report.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return report
