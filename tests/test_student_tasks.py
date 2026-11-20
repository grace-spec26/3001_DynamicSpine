from pathlib import Path

import numpy as np
import pytest

import student_tasks as tasks


def test_centroid_affine_and_landmark_baselines() -> None:
    mask = np.zeros((5, 7, 9), dtype=np.uint8)
    mask[1:3, 2:4, 4:6] = 8
    np.testing.assert_allclose(tasks.label_centroid_ijk(mask, 8), [1.5, 2.5, 4.5])
    with pytest.raises(ValueError):
        tasks.label_centroid_ijk(mask, 9)

    affine = np.array(
        [[0, -2, 0, 10], [3, 0, 0, -2], [0, 0, 4, 1], [0, 0, 0, 1]]
    )
    np.testing.assert_allclose(
        tasks.apply_affine(np.array([[1, 2, 3], [0, 0, 0]]), affine),
        [[6, 1, 13], [10, -2, 1]],
    )

    centres = np.arange(90).reshape(2, 15, 3)
    np.testing.assert_array_equal(
        tasks.landmark_features(centres), np.arange(90).reshape(2, 45)
    )


def test_mesh_row_uses_anatomical_order_and_relative_paths(tmp_path: Path) -> None:
    root = tmp_path / "package"
    output = root / "outputs"
    output.mkdir(parents=True)
    levels = [f"T{i}" for i in range(1, 13)] + ["L1", "L2", "L3"]
    mesh_dir = root / "data" / "meshes" / "case-a"
    mesh_dir.mkdir(parents=True)
    for level in levels:
        (mesh_dir / f"{level}.ply").write_text("ply\n", encoding="utf-8")

    row = tasks.mesh_row("case-a", root, output)

    assert row[:2] == ["case-a", ""]
    assert [Path(value).stem for value in row[2:]] == levels
    assert all((output / value).is_file() for value in row[2:])


def test_pls_accepts_single_feature_used_by_exercise_checker() -> None:
    x = np.arange(20, dtype=float).reshape(-1, 1)
    y = np.column_stack((2 * x[:, 0] + 3, -x[:, 0]))

    model = tasks.fit_pls(x, y, 1)
    prediction = tasks.predict_scores(model, x)

    assert prediction.shape == (20, 2)
    np.testing.assert_allclose(prediction, y, atol=1e-5)


@pytest.mark.parametrize("n_components", [1, 2, 3, 5, 8])
def test_pls_accepts_course_component_grid(n_components: int) -> None:
    rng = np.random.default_rng(3001)
    x = rng.normal(size=(30, 45))
    y = x[:, :3] @ np.array(
        [[1.0, -0.5, 0.25], [0.0, 1.5, -0.75], [0.5, 0.25, 1.0]]
    )

    model = tasks.fit_pls(x, y, n_components)

    assert model.n_components == n_components
    assert tasks.predict_scores(model, x[:4]).shape == (4, 3)


@pytest.mark.parametrize(
    ("x", "y", "message"),
    [
        (np.ones((3, 2)), np.ones((2, 1)), "same number of rows"),
        (np.array([[1.0, np.nan], [2.0, 3.0]]), np.ones((2, 1)), "finite"),
        (np.ones((2, 2)), np.array([[1.0], [np.inf]]), "finite"),
    ],
)
def test_pls_rejects_incompatible_or_nonfinite_training_data(
    x: np.ndarray, y: np.ndarray, message: str
) -> None:
    with pytest.raises(ValueError, match=message):
        tasks.fit_pls(x, y, 1)


def test_predict_scores_rejects_wrong_feature_count_and_nonfinite_values() -> None:
    x = np.arange(20, dtype=float).reshape(-1, 1)
    y = np.column_stack((x[:, 0], -x[:, 0]))
    model = tasks.fit_pls(x, y, 1)

    with pytest.raises(ValueError, match="feature columns"):
        tasks.predict_scores(model, np.ones((2, 2)))
    with pytest.raises(ValueError, match="finite"):
        tasks.predict_scores(model, np.array([[np.nan]]))


def test_reconstruct_particles_validates_shapes_and_reconstructs_mean() -> None:
    mean = np.zeros((4, 3))
    modes = np.zeros((2, 12))
    modes[0, 0] = 1
    modes[1, 1] = 1
    scores = np.array([[0.0, 0.0], [2.0, -3.0]])

    result = tasks.reconstruct_particles(scores, mean, modes)

    assert result.shape == (2, 4, 3)
    np.testing.assert_allclose(result[0], mean)
    np.testing.assert_allclose(result[1, 0], [2, -3, 0])


@pytest.mark.parametrize(
    ("scores", "mean", "modes"),
    [
        (np.ones(2), np.zeros((4, 3)), np.zeros((2, 12))),
        (np.ones((1, 2)), np.zeros((4, 2)), np.zeros((2, 8))),
        (np.ones((1, 2)), np.zeros((4, 3)), np.zeros((3, 12))),
        (np.ones((1, 2)), np.zeros((4, 3)), np.zeros((2, 11))),
        (np.array([[np.nan, 0.0]]), np.zeros((4, 3)), np.zeros((2, 12))),
    ],
)
def test_reconstruct_particles_rejects_invalid_inputs(
    scores: np.ndarray, mean: np.ndarray, modes: np.ndarray
) -> None:
    with pytest.raises(ValueError):
        tasks.reconstruct_particles(scores, mean, modes)
