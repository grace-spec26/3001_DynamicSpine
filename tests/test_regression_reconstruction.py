from pathlib import Path

import numpy as np
import pytest

from reconstruction_pipeline import build_reconstructions, write_reconstructions
from regression_pipeline import fit_regression


def _regression_fixture() -> tuple[np.ndarray, np.ndarray, list[str], dict[str, str]]:
    rng = np.random.default_rng(3001)
    features = rng.normal(size=(30, 12))
    scores = features[:, :6] @ rng.normal(size=(6, 6))
    case_ids = [f"case-{index:02d}" for index in range(30)]
    split = {
        case_id: ("test" if index in {2, 7, 13, 21, 29} else "train")
        for index, case_id in enumerate(case_ids)
    }
    return features, scores, case_ids, split


def test_regression_keeps_test_cases_out_of_training_and_preserves_case_order() -> None:
    features, scores, case_ids, split = _regression_fixture()

    result = fit_regression(features, scores, case_ids, split)

    expected_test = [case_id for case_id in case_ids if split[case_id] == "test"]
    expected_train = [case_id for case_id in case_ids if split[case_id] == "train"]
    assert result["test_case_ids"] == expected_test
    assert result["training_case_ids"] == expected_train
    assert result["predicted"].shape == (5, 6)
    assert result["actual"].shape == (5, 6)
    assert [trial["pls_components"] for trial in result["cv"]] == [1, 2, 3, 5, 8]
    assert result["selected"]["pls_components"] in {1, 2, 3, 5, 8}


def test_test_scores_do_not_influence_fitted_predictions() -> None:
    features, scores, case_ids, split = _regression_fixture()
    changed_scores = scores.copy()
    test_indices = [index for index, case_id in enumerate(case_ids) if split[case_id] == "test"]
    changed_scores[test_indices] += 10000

    original = fit_regression(features, scores, case_ids, split)
    changed = fit_regression(features, changed_scores, case_ids, split)

    np.testing.assert_allclose(original["predicted"], changed["predicted"])
    assert original["selected"] == changed["selected"]
    assert original["cv"] == changed["cv"]
    assert not np.array_equal(original["actual"], changed["actual"])


@pytest.mark.parametrize(
    ("features", "scores", "message"),
    [
        (np.ones((4, 3)), np.ones((3, 2)), "rows"),
        (np.array([[1.0, np.nan], [2.0, 3.0]]), np.ones((2, 1)), "finite"),
        (np.ones((2, 2)), np.array([[1.0], [np.inf]]), "finite"),
    ],
)
def test_regression_rejects_mismatched_or_nonfinite_arrays(
    features: np.ndarray, scores: np.ndarray, message: str
) -> None:
    case_ids = [f"case-{index}" for index in range(features.shape[0])]
    split = {case_id: "train" for case_id in case_ids}
    if case_ids:
        split[case_ids[-1]] = "test"
    with pytest.raises(ValueError, match=message):
        fit_regression(features, scores, case_ids, split)


def test_reconstruction_builds_3840_particles_and_fifteen_ordered_domains(
    tmp_path: Path,
) -> None:
    rng = np.random.default_rng(3001)
    case_ids = ["case-a", "case-b"]
    domains = [f"T{i}" for i in range(1, 13)] + ["L1", "L2", "L3"]
    mean = np.zeros((3840, 3))
    modes = rng.normal(size=(3, 11520))
    modes, _ = np.linalg.qr(modes.T)
    modes = modes.T
    predicted_scores = np.array([[1.0, 2.0, 3.0], [-1.0, 0.5, 2.0]])
    actual_scores = np.zeros((2, 3))

    result = build_reconstructions(
        case_ids,
        predicted_scores,
        actual_scores,
        mean,
        modes,
        domains,
        particles_per_domain=256,
    )
    write_reconstructions(tmp_path, result, domains, particles_per_domain=256)

    assert result["predicted"].shape == (2, 3840, 3)
    assert result["actual"].shape == (2, 3840, 3)
    assert list(result["domains"]["case-a"]) == domains
    for case_id in case_ids:
        files = list((tmp_path / "particles" / case_id).glob("*.particles"))
        assert len(files) == 15
        assert np.loadtxt(tmp_path / "particles" / case_id / "T1.particles").shape == (
            256,
            3,
        )


def test_reconstruction_rejects_nonfinite_scores() -> None:
    with pytest.raises(ValueError, match="finite"):
        build_reconstructions(
            ["case-a"],
            np.array([[np.nan]]),
            np.zeros((1, 1)),
            np.zeros((2, 3)),
            np.zeros((1, 6)),
            ["T1"],
            particles_per_domain=2,
        )
