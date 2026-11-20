"""Validate standardized PCA exports and pack a compact NumPy bundle."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd


def pack_pca_exports(folder: Path, output_path: Path, mode_count: int) -> Path:
    """Validate *mode_count* PCA modes and save ``pca.npz``."""
    folder = Path(folder)
    output_path = Path(output_path)
    if mode_count < 1:
        raise ValueError("Use at least one PCA mode")
    metadata = json.loads((folder / "metadata.json").read_text(encoding="utf-8"))
    domains = list(metadata["domains"])
    particles_per_domain = int(metadata["particles_per_domain"])
    mean = np.vstack(
        [np.loadtxt(folder / "mean" / f"{domain}.particles") for domain in domains]
    )
    modes = np.stack(
        [
            np.loadtxt(folder / "modes" / f"pc{index}.txt").reshape(-1)
            for index in range(1, mode_count + 1)
        ]
    )
    table = pd.read_csv(folder / "scores.csv", dtype={"case_id": str})
    case_ids = table["case_id"].to_numpy(dtype=str)
    if case_ids.tolist() != list(map(str, metadata.get("case_ids", []))):
        raise ValueError("scores.csv case order does not match metadata case order")
    scores = table[
        [f"pc{index}" for index in range(1, mode_count + 1)]
    ].to_numpy(dtype=float)
    eigenvalues = np.atleast_1d(
        np.loadtxt(folder / "eigenvalues.txt", dtype=float)
    )[:mode_count]
    coordinate_count = len(domains) * particles_per_domain * 3
    if mean.shape != (len(domains) * particles_per_domain, 3):
        raise ValueError("mean particles do not match metadata")
    if modes.shape != (mode_count, coordinate_count):
        raise ValueError("PCA modes do not match metadata")
    if eigenvalues.shape != (mode_count,):
        raise ValueError("not enough PCA eigenvalues")
    if len(set(case_ids)) != len(case_ids) or len(case_ids) <= mode_count:
        raise ValueError("Case IDs must be unique; use fewer modes than subjects")
    if not all(
        np.isfinite(value).all() for value in (mean, modes, scores, eigenvalues)
    ):
        raise ValueError("Non-finite PCA export values")
    if not np.allclose(modes @ modes.T, np.eye(mode_count), atol=1e-8):
        raise ValueError("Expected orthonormal unit eigenvectors")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez(
        output_path,
        mean=mean,
        modes=modes,
        scores=scores,
        eigenvalues=eigenvalues,
        case_ids=case_ids,
    )
    return output_path
