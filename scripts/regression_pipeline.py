"""Cross-validated PLS regression over a fixed course train/test split."""

from __future__ import annotations

from typing import Any

import numpy as np
from sklearn.model_selection import KFold

from student_tasks import fit_pls, predict_scores


def fit_regression(
    features: np.ndarray,
    scores: np.ndarray,
    case_ids: list[str],
    split_by_id: dict[str, str],
    candidate_components: tuple[int, ...] = (1, 2, 3, 5, 8),
    folds: int = 3,
    random_state: int = 42,
) -> dict[str, Any]:
    """Fit training-only PLS CV and predict the held-out cases."""
    features = np.asarray(features, dtype=float)
    scores = np.asarray(scores, dtype=float)
    case_ids = [str(case_id) for case_id in case_ids]
    if features.ndim != 2 or scores.ndim != 2:
        raise ValueError("features and scores must both be 2D")
    if features.shape[0] != scores.shape[0] or features.shape[0] != len(case_ids):
        raise ValueError("features, scores, and case IDs must have matching rows")
    if not np.isfinite(features).all() or not np.isfinite(scores).all():
        raise ValueError("features and scores must contain only finite values")
    if len(set(case_ids)) != len(case_ids):
        raise ValueError("case IDs must be unique")
    unknown = [case_id for case_id in case_ids if case_id not in split_by_id]
    if unknown:
        raise ValueError("split is missing case IDs: " + ", ".join(unknown))
    invalid_splits = sorted(
        {split_by_id[case_id] for case_id in case_ids}.difference({"train", "test"})
    )
    if invalid_splits:
        raise ValueError("invalid split values: " + ", ".join(invalid_splits))

    train_mask = np.array([split_by_id[case_id] == "train" for case_id in case_ids])
    test_mask = ~train_mask
    if train_mask.sum() < max(6, folds) or test_mask.sum() < 1:
        raise ValueError("split requires at least six training and one test case")
    train_features = features[train_mask]
    train_scores = scores[train_mask]
    cv_splits = list(
        KFold(n_splits=folds, shuffle=True, random_state=random_state).split(
            train_features
        )
    )
    maximum = min(
        min(len(train_indices) for train_indices, _ in cv_splits),
        features.shape[1],
    )
    candidates = [
        int(value)
        for value in candidate_components
        if int(value) >= 1 and int(value) <= maximum
    ]
    if not candidates:
        raise ValueError("no supported PLS component candidates")

    trials: list[dict[str, float | int]] = []
    for component_count in candidates:
        fold_mse: list[float] = []
        for train_indices, validation_indices in cv_splits:
            model = fit_pls(
                train_features[train_indices],
                train_scores[train_indices],
                component_count,
            )
            predicted = np.asarray(
                predict_scores(model, train_features[validation_indices]), dtype=float
            )
            if predicted.shape != train_scores[validation_indices].shape:
                raise ValueError("prediction shape mismatch")
            fold_mse.append(
                float(np.mean((predicted - train_scores[validation_indices]) ** 2))
            )
        trials.append(
            {
                "pls_components": component_count,
                "cv_rmse": float(np.sqrt(np.mean(fold_mse))),
            }
        )
    selected = min(trials, key=lambda trial: float(trial["cv_rmse"]))
    model = fit_pls(
        train_features, train_scores, int(selected["pls_components"])
    )
    predicted = np.asarray(predict_scores(model, features[test_mask]), dtype=float)
    actual = scores[test_mask]
    if predicted.shape != actual.shape:
        raise ValueError("prediction shape mismatch")
    baseline = np.broadcast_to(train_scores.mean(axis=0), actual.shape)
    return {
        "model": model,
        "cv": trials,
        "selected": selected,
        "training_case_ids": [
            case_id for case_id, keep in zip(case_ids, train_mask) if keep
        ],
        "test_case_ids": [
            case_id for case_id, keep in zip(case_ids, test_mask) if keep
        ],
        "predicted": predicted,
        "actual": actual,
        "test_rmse": float(np.sqrt(np.mean((predicted - actual) ** 2))),
        "mean_score_baseline_rmse": float(
            np.sqrt(np.mean((baseline - actual) ** 2))
        ),
    }
