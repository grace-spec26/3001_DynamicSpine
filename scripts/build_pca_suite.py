"""Build K=3,4,5,6 PCA/PLS/reconstruction experiments for the viewer."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Any

import joblib
import numpy as np

from common import LEVELS, OUTPUT, ROOT
from reconstruction_pipeline import build_reconstructions, write_reconstructions
from regression_pipeline import fit_regression
from student_tasks import landmark_features


def _case_metadata(root: Path) -> tuple[dict[str, str], dict[str, np.ndarray]]:
    with (root / "data" / "cases.csv").open(
        newline="", encoding="utf-8"
    ) as handle:
        rows = list(csv.DictReader(handle))
    split = {row["case_id"]: row["regression_split"] for row in rows}
    centres = {
        row["case_id"]: np.asarray(
            json.loads(
                (root / "outputs" / "centroids" / f"{row['case_id']}.json").read_text(
                    encoding="utf-8"
                )
            )["aligned_mm"],
            dtype=float,
        )
        for row in rows
    }
    return split, centres


def build_suite(
    root: Path,
    pca_path: Path,
    output_dir: Path,
    mode_counts: tuple[int, ...] = (3, 4, 5, 6),
) -> list[dict[str, Any]]:
    """Build validated experiment folders and return their summary rows."""
    root = Path(root)
    output_dir = Path(output_dir)
    with np.load(pca_path, allow_pickle=False) as source:
        mean = np.asarray(source["mean"], dtype=float)
        all_modes = np.asarray(source["modes"], dtype=float)
        all_scores = np.asarray(source["scores"], dtype=float)
        eigenvalues = np.asarray(source["eigenvalues"], dtype=float)
        case_ids = source["case_ids"].astype(str).tolist()
    split, centres_by_id = _case_metadata(root)
    if any(case_id not in split or case_id not in centres_by_id for case_id in case_ids):
        raise ValueError("PCA case IDs do not match cohort metadata")
    centres = np.stack([centres_by_id[case_id] for case_id in case_ids])
    if centres.shape != (len(case_ids), 15, 3) or not np.isfinite(centres).all():
        raise ValueError("expected finite 15-domain centroids for every PCA case")
    features = landmark_features(centres)
    summaries: list[dict[str, Any]] = []

    for mode_count in mode_counts:
        if mode_count < 1 or mode_count > len(all_modes):
            raise ValueError(f"unsupported PCA mode count: {mode_count}")
        folder = output_dir / f"pca_{mode_count:02d}"
        folder.mkdir(parents=True, exist_ok=True)
        modes = all_modes[:mode_count]
        scores = all_scores[:, :mode_count]
        selected_eigenvalues = eigenvalues[:mode_count]
        np.savez(
            folder / "pca.npz",
            mean=mean,
            modes=modes,
            scores=scores,
            eigenvalues=selected_eigenvalues,
            case_ids=np.asarray(case_ids, dtype=str),
        )
        regression = fit_regression(features, scores, case_ids, split)
        report = {
            "pca_modes": mode_count,
            "cv": regression["cv"],
            "selected": regression["selected"],
            "test_rmse": regression["test_rmse"],
            "mean_score_baseline_rmse": regression["mean_score_baseline_rmse"],
            "training_cases": regression["training_case_ids"],
            "test_cases": regression["test_case_ids"],
            "scope": "Regression holdout within a shared SSM; not external pipeline validation",
        }
        (folder / "regression_report.json").write_text(
            json.dumps(report, indent=2) + "\n", encoding="utf-8"
        )
        np.savez(
            folder / "predicted_scores.npz",
            case_ids=np.asarray(regression["test_case_ids"], dtype=str),
            predicted=regression["predicted"],
            actual=regression["actual"],
        )
        joblib.dump(
            {
                "model": regression["model"],
                "mean": mean,
                "modes": modes,
                "eigenvalues": selected_eigenvalues,
                "case_ids": np.asarray(case_ids, dtype=str),
                "centres": centres,
                "default_case": regression["test_case_ids"][0],
                "schema_version": 2,
            },
            folder / "interactive_model.joblib",
        )
        reconstructions = build_reconstructions(
            regression["test_case_ids"],
            regression["predicted"],
            regression["actual"],
            mean,
            modes,
            list(LEVELS),
            particles_per_domain=256,
        )
        write_reconstructions(
            folder, reconstructions, list(LEVELS), particles_per_domain=256
        )
        summaries.append(
            {
                "pca_modes": mode_count,
                "pls_components": regression["selected"]["pls_components"],
                "test_rmse": regression["test_rmse"],
                "baseline_rmse": regression["mean_score_baseline_rmse"],
                "mean_particle_error_mm": float(
                    np.mean(
                        [
                            error["mean_particle_error_mm"]
                            for error in reconstructions["errors"]
                        ]
                    )
                ),
            }
        )
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "summary.json").write_text(
        json.dumps(summaries, indent=2) + "\n", encoding="utf-8"
    )
    return summaries


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--pca", type=Path, default=OUTPUT / "full_pca.npz")
    parser.add_argument("--output", type=Path, default=OUTPUT / "pca_models")
    args = parser.parse_args(argv)
    summaries = build_suite(ROOT, args.pca, args.output)
    for summary in summaries:
        print(
            "K={pca_modes}: PLS={pls_components}, test_rmse={test_rmse:.6f}, "
            "particle_error={mean_particle_error_mm:.6f} mm".format(**summary)
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
