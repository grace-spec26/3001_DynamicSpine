"""Shared strict validation for interactive PLS/PCA model bundles."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import joblib
import numpy as np

from student_tasks import landmark_features, predict_scores, reconstruct_particles


def _require_finite(name: str, value: Any) -> np.ndarray:
    array = np.asarray(value)
    if not np.issubdtype(array.dtype, np.number) or not np.isfinite(array).all():
        raise ValueError(f"{name} must contain only finite numeric values")
    return array


def load_validated_model_bundle(
    model_path: Path, pca_path: Path | None = None
) -> dict[str, Any]:
    """Load a model only after semantic validation and a real prediction smoke test."""
    data = joblib.load(Path(model_path))
    if not isinstance(data, dict) or data.get("schema_version") not in {1, 2}:
        raise ValueError("unknown model schema")
    model = data["model"]
    mean = _require_finite("mean", data["mean"]).astype(float)
    modes = _require_finite("modes", data["modes"]).astype(float)
    eigenvalues = _require_finite(
        "eigenvalues", data.get("eigenvalues", np.ones(len(modes)))
    ).astype(float)
    centres = _require_finite("centres", data["centres"]).astype(float)
    case_ids = list(map(str, data["case_ids"]))
    default_case = str(data["default_case"])
    if (
        mean.shape != (3840, 3)
        or modes.ndim != 2
        or modes.shape[0] < 1
        or modes.shape[1] != 11520
        or eigenvalues.shape != (len(modes),)
        or centres.shape != (len(case_ids), 15, 3)
        or len(set(case_ids)) != len(case_ids)
        or default_case not in case_ids
        or getattr(model, "n_features_in_", None) != 45
    ):
        raise ValueError("inconsistent model dimensions or identifiers")

    for attribute in (
        "coef_",
        "intercept_",
        "x_weights_",
        "y_weights_",
        "x_loadings_",
        "y_loadings_",
        "x_rotations_",
        "y_rotations_",
    ):
        if hasattr(model, attribute):
            _require_finite(f"estimator.{attribute}", getattr(model, attribute))

    if pca_path is not None:
        with np.load(Path(pca_path), allow_pickle=False) as pca:
            required = {"mean", "modes", "eigenvalues", "case_ids", "scores"}
            if not required.issubset(pca.files):
                raise ValueError("companion PCA bundle is incomplete")
            comparisons = {
                "mean": mean,
                "modes": modes,
                "eigenvalues": eigenvalues,
            }
            for name, expected in comparisons.items():
                actual = _require_finite(f"pca.{name}", pca[name])
                if actual.shape != expected.shape or not np.allclose(actual, expected):
                    raise ValueError(f"model and PCA {name} do not match")
            if pca["case_ids"].astype(str).tolist() != case_ids:
                raise ValueError("model and PCA case order do not match")
            scores = _require_finite("pca.scores", pca["scores"])
            if scores.shape != (len(case_ids), len(modes)):
                raise ValueError("companion PCA scores have the wrong shape")

    index = case_ids.index(default_case)
    features = landmark_features(centres[index : index + 1])
    scores = np.asarray(predict_scores(model, features), dtype=float)
    if scores.shape != (1, len(modes)) or not np.isfinite(scores).all():
        raise ValueError("estimator prediction smoke test failed")
    particles = np.asarray(reconstruct_particles(scores, mean, modes), dtype=float)
    if particles.shape != (1, 3840, 3) or not np.isfinite(particles).all():
        raise ValueError("reconstruction smoke test failed")

    result = dict(data)
    result.update(
        mean=mean,
        modes=modes,
        eigenvalues=eigenvalues,
        centres=centres,
        case_ids=case_ids,
        default_case=default_case,
    )
    return result
