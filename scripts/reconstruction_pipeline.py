"""Reconstruct predicted PCA scores and split spines into ordered domains."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np

from student_tasks import reconstruct_particles


def build_reconstructions(
    case_ids: list[str],
    predicted_scores: np.ndarray,
    actual_scores: np.ndarray,
    mean: np.ndarray,
    modes: np.ndarray,
    domains: list[str],
    particles_per_domain: int = 256,
) -> dict[str, Any]:
    """Reconstruct and return concatenated and per-domain particle arrays."""
    predicted_scores = np.asarray(predicted_scores, dtype=float)
    actual_scores = np.asarray(actual_scores, dtype=float)
    mean = np.asarray(mean, dtype=float)
    modes = np.asarray(modes, dtype=float)
    case_ids = [str(case_id) for case_id in case_ids]
    if predicted_scores.shape != actual_scores.shape:
        raise ValueError("predicted and actual scores must have matching shapes")
    if predicted_scores.ndim != 2 or predicted_scores.shape[0] != len(case_ids):
        raise ValueError("score rows must match case IDs")
    if not all(
        np.isfinite(value).all()
        for value in (predicted_scores, actual_scores, mean, modes)
    ):
        raise ValueError("reconstruction inputs must contain only finite values")
    expected_points = len(domains) * particles_per_domain
    if mean.shape != (expected_points, 3):
        raise ValueError(
            f"mean must have ({expected_points}, 3) points for the requested domains"
        )
    predicted = reconstruct_particles(predicted_scores, mean, modes)
    actual = reconstruct_particles(actual_scores, mean, modes)
    domain_map: dict[str, dict[str, np.ndarray]] = {}
    errors: list[dict[str, float | str]] = []
    for case_id, points, reference in zip(case_ids, predicted, actual):
        domain_map[case_id] = {
            domain: points[
                index * particles_per_domain : (index + 1) * particles_per_domain
            ]
            for index, domain in enumerate(domains)
        }
        errors.append(
            {
                "case_id": case_id,
                "mean_particle_error_mm": float(
                    np.linalg.norm(points - reference, axis=1).mean()
                ),
            }
        )
    return {
        "case_ids": case_ids,
        "predicted": predicted,
        "actual": actual,
        "domains": domain_map,
        "errors": errors,
    }


def write_reconstructions(
    output_dir: Path,
    result: dict[str, Any],
    domains: list[str],
    particles_per_domain: int = 256,
) -> None:
    """Persist reconstruction bundle, per-case domains, and error report."""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    np.savez(
        output_dir / "reconstructions.npz",
        case_ids=np.asarray(result["case_ids"], dtype=str),
        predicted=result["predicted"],
        actual=result["actual"],
    )
    particle_root = output_dir / "particles"
    for case_id in result["case_ids"]:
        case_dir = particle_root / case_id
        case_dir.mkdir(parents=True, exist_ok=True)
        for domain in domains:
            values = result["domains"][case_id][domain]
            if values.shape != (particles_per_domain, 3):
                raise ValueError(f"invalid domain particle shape for {case_id}/{domain}")
            np.savetxt(case_dir / f"{domain}.particles", values, fmt="%.8f")
    (output_dir / "particle_errors.json").write_text(
        json.dumps(result["errors"], indent=2) + "\n", encoding="utf-8"
    )
