"""Compute and export whole-spine PCA from validated world particles."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from common import LEVELS, OUTPUT, ROOT, output_relative_path
from shapeworks_completion import validate_particle_store


def load_particle_matrix(
    particle_root: Path,
    case_ids: list[str],
    domains: list[str],
    particles_per_domain: int = 256,
    coordinate_system: str = "world",
    expected_case_ids: list[str] | None = None,
) -> np.ndarray:
    """Load a deterministic case-major matrix of concatenated spine particles."""
    if expected_case_ids is not None and list(case_ids) != list(expected_case_ids):
        raise ValueError("case order does not match the manifest")
    if coordinate_system not in {"local", "world"}:
        raise ValueError("coordinate_system must be 'local' or 'world'")
    rows: list[np.ndarray] = []
    for case_id in case_ids:
        domain_points: list[np.ndarray] = []
        for domain in domains:
            path = (
                Path(particle_root)
                / f"{case_id}_{domain}_{coordinate_system}.particles"
            )
            if not path.is_file():
                raise ValueError(f"missing particle file: {path}")
            try:
                values = np.atleast_2d(np.loadtxt(path, dtype=float))
            except (OSError, ValueError) as exc:
                raise ValueError(f"malformed particle file {path}: {exc}") from exc
            if values.shape != (particles_per_domain, 3):
                raise ValueError(
                    f"{path}: expected ({particles_per_domain}, 3), got {values.shape}"
                )
            if not np.isfinite(values).all():
                raise ValueError(f"{path}: particle values must be finite")
            domain_points.append(values)
        rows.append(np.vstack(domain_points).reshape(-1))
    matrix = np.vstack(rows)
    if not np.isfinite(matrix).all():
        raise ValueError("particle matrix must be finite")
    return matrix


def load_particle_store_matrix(
    store_path: Path,
    case_ids: list[str],
    domains: list[str],
    particles_per_domain: int = 256,
    coordinate_system: str = "world",
) -> np.ndarray:
    """Load the durable, ordered particle store used for analysis."""
    if coordinate_system not in {"local", "world"}:
        raise ValueError("coordinate_system must be 'local' or 'world'")
    bundle = validate_particle_store(
        store_path, case_ids, domains, particles_per_domain=particles_per_domain
    )
    return bundle[coordinate_system].reshape(len(case_ids), -1)


def compute_pca(matrix: np.ndarray) -> dict[str, np.ndarray]:
    """Return mean, unit modes, raw projection scores, and sample eigenvalues."""
    matrix = np.asarray(matrix, dtype=float)
    if matrix.ndim != 2 or matrix.shape[0] < 2 or matrix.shape[1] < 1:
        raise ValueError("matrix must have shape (subjects >= 2, coordinates >= 1)")
    if not np.isfinite(matrix).all():
        raise ValueError("matrix must contain only finite values")
    mean = matrix.mean(axis=0)
    centered = matrix - mean
    _, singular_values, modes = np.linalg.svd(centered, full_matrices=False)
    component_count = min(matrix.shape[0] - 1, matrix.shape[1])
    modes = modes[:component_count]
    singular_values = singular_values[:component_count]
    scores = centered @ modes.T
    eigenvalues = singular_values**2 / (matrix.shape[0] - 1)
    return {
        "mean": mean,
        "modes": modes,
        "scores": scores,
        "eigenvalues": eigenvalues,
    }


def write_pca_exports(
    folder: Path,
    case_ids: list[str],
    domains: list[str],
    particles_per_domain: int,
    mean: np.ndarray,
    modes: np.ndarray,
    eigenvalues: np.ndarray,
    scores: np.ndarray,
) -> None:
    """Write the standardized mean/modes/scores/eigenvalues export layout."""
    folder = Path(folder)
    mean = np.asarray(mean, dtype=float)
    modes = np.asarray(modes, dtype=float)
    eigenvalues = np.asarray(eigenvalues, dtype=float)
    scores = np.asarray(scores, dtype=float)
    coordinate_count = len(domains) * particles_per_domain * 3
    if mean.shape != (coordinate_count,):
        raise ValueError("mean size does not match domains and particles_per_domain")
    if modes.ndim != 2 or modes.shape[1] != coordinate_count:
        raise ValueError("modes do not match the mean coordinate count")
    if eigenvalues.shape != (modes.shape[0],):
        raise ValueError("eigenvalues must match the number of modes")
    if scores.shape != (len(case_ids), modes.shape[0]):
        raise ValueError("scores must match case and mode counts")
    if not all(np.isfinite(value).all() for value in (mean, modes, eigenvalues, scores)):
        raise ValueError("PCA exports must be finite")
    if not np.allclose(modes @ modes.T, np.eye(len(modes)), atol=1e-8):
        raise ValueError("PCA modes must be orthonormal unit eigenvectors")

    mean_dir = folder / "mean"
    modes_dir = folder / "modes"
    mean_dir.mkdir(parents=True, exist_ok=True)
    modes_dir.mkdir(parents=True, exist_ok=True)
    mean_points = mean.reshape(len(domains), particles_per_domain, 3)
    for domain, values in zip(domains, mean_points):
        np.savetxt(mean_dir / f"{domain}.particles", values)
    for index, mode in enumerate(modes, start=1):
        np.savetxt(modes_dir / f"pc{index}.txt", mode)
    table = pd.DataFrame({"case_id": case_ids})
    for index in range(modes.shape[0]):
        table[f"pc{index + 1}"] = scores[:, index]
    table.to_csv(folder / "scores.csv", index=False)
    np.savetxt(folder / "eigenvalues.txt", eigenvalues)
    (folder / "metadata.json").write_text(
        json.dumps(
            {
                "case_ids": case_ids,
                "domains": domains,
                "particles_per_domain": particles_per_domain,
                "coordinate_system": "world",
                "score_convention": "raw_projection_coefficients",
                "mode_convention": "unit_eigenvectors",
            },
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )


def _manifest_case_ids(root: Path) -> list[str]:
    with (root / "data" / "cases.csv").open(newline="", encoding="utf-8") as handle:
        return [row["case_id"] for row in csv.DictReader(handle)]


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--particles", type=Path, default=OUTPUT / "full_run_particles"
    )
    parser.add_argument(
        "--particle-store", type=Path, default=OUTPUT / "particle_store.npz"
    )
    parser.add_argument("--output", type=Path, default=OUTPUT / "pca_exports")
    args = parser.parse_args(argv)
    case_ids = _manifest_case_ids(ROOT)
    if args.particle_store.is_file():
        matrix = load_particle_store_matrix(
            args.particle_store, case_ids, list(LEVELS), particles_per_domain=256
        )
    else:
        matrix = load_particle_matrix(
            args.particles,
            case_ids,
            list(LEVELS),
            particles_per_domain=256,
            coordinate_system="world",
            expected_case_ids=case_ids,
        )
    pca = compute_pca(matrix)
    write_pca_exports(
        args.output,
        case_ids,
        list(LEVELS),
        256,
        pca["mean"],
        pca["modes"],
        pca["eigenvalues"],
        pca["scores"],
    )
    print(
        f"Prepared {len(case_ids)} subjects and {len(pca['eigenvalues'])} PCA modes"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
