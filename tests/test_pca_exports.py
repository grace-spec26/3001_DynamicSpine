from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from import_pca_exports import pack_pca_exports
from prepare_full_pca_exports import (
    compute_pca,
    load_particle_matrix,
    output_relative_path,
    write_pca_exports,
)


def _particle_file(path: Path, values: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savetxt(path, values)


def test_output_relative_path_normalizes_windows_posix_and_mixed_separators() -> None:
    assert output_relative_path("particles/case_T1_world.particles").parts[-2:] == (
        "particles",
        "case_T1_world.particles",
    )
    assert output_relative_path(r"particles\case_T1_world.particles").parts[-2:] == (
        "particles",
        "case_T1_world.particles",
    )
    assert output_relative_path(r"particles\case/T1_world.particles").parts[-3:] == (
        "particles",
        "case",
        "T1_world.particles",
    )


def test_load_particle_matrix_uses_given_case_and_domain_order(tmp_path: Path) -> None:
    root = tmp_path / "particles"
    values = {
        ("case-a", "T1"): np.array([[1, 2, 3], [4, 5, 6]], dtype=float),
        ("case-a", "L1"): np.array([[7, 8, 9], [10, 11, 12]], dtype=float),
        ("case-b", "T1"): np.array([[13, 14, 15], [16, 17, 18]], dtype=float),
        ("case-b", "L1"): np.array([[19, 20, 21], [22, 23, 24]], dtype=float),
    }
    for (case_id, domain), points in values.items():
        _particle_file(root / f"{case_id}_{domain}_world.particles", points)

    matrix = load_particle_matrix(
        root,
        ["case-b", "case-a"],
        ["L1", "T1"],
        particles_per_domain=2,
        expected_case_ids=["case-b", "case-a"],
    )

    np.testing.assert_array_equal(
        matrix[0].reshape(4, 3),
        np.vstack((values[("case-b", "L1")], values[("case-b", "T1")])),
    )


def test_load_particle_matrix_rejects_shuffled_case_order(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="case order"):
        load_particle_matrix(
            tmp_path,
            ["case-b", "case-a"],
            ["T1"],
            particles_per_domain=2,
            expected_case_ids=["case-a", "case-b"],
        )


def test_load_particle_matrix_rejects_nonfinite_particles(tmp_path: Path) -> None:
    root = tmp_path / "particles"
    _particle_file(
        root / "case-a_T1_world.particles",
        np.array([[0.0, 1.0, 2.0], [3.0, np.nan, 5.0]]),
    )

    with pytest.raises(ValueError, match="finite"):
        load_particle_matrix(
            root,
            ["case-a"],
            ["T1"],
            particles_per_domain=2,
            expected_case_ids=["case-a"],
        )


def test_compute_pca_returns_orthonormal_modes_and_reconstructs_training_data() -> None:
    matrix = np.array(
        [
            [0.0, 0.0, 0.0, 0.0],
            [1.0, 0.0, 1.0, 0.0],
            [0.0, 2.0, 0.0, 2.0],
            [2.0, 1.0, 2.0, 1.0],
        ]
    )

    result = compute_pca(matrix)

    np.testing.assert_allclose(
        result["modes"] @ result["modes"].T,
        np.eye(result["modes"].shape[0]),
        atol=1e-12,
    )
    reconstructed = result["scores"] @ result["modes"] + result["mean"]
    np.testing.assert_allclose(reconstructed, matrix, atol=1e-12)
    assert np.all(result["eigenvalues"][:-1] >= result["eigenvalues"][1:])


def test_write_and_pack_pca_exports_preserve_raw_scores_and_eigenvalues(
    tmp_path: Path,
) -> None:
    rng = np.random.default_rng(3001)
    matrix = rng.normal(size=(8, 12))
    pca = compute_pca(matrix)
    folder = tmp_path / "pca_exports"

    write_pca_exports(
        folder,
        [f"case-{index}" for index in range(8)],
        ["T1", "L1"],
        particles_per_domain=2,
        mean=pca["mean"],
        modes=pca["modes"],
        eigenvalues=pca["eigenvalues"],
        scores=pca["scores"],
    )
    output = tmp_path / "pca.npz"
    pack_pca_exports(folder, output, mode_count=4)

    bundle = np.load(output)
    assert set(bundle.files) == {
        "case_ids",
        "mean",
        "modes",
        "eigenvalues",
        "scores",
    }
    assert bundle["mean"].shape == (4, 3)
    assert bundle["modes"].shape == (4, 12)
    np.testing.assert_allclose(bundle["scores"], pca["scores"][:, :4])
    np.testing.assert_allclose(bundle["eigenvalues"], pca["eigenvalues"][:4])
    table = pd.read_csv(folder / "scores.csv")
    assert table.columns[:5].tolist() == ["case_id", "pc1", "pc2", "pc3", "pc4"]


def test_pack_rejects_scores_shuffled_from_metadata_case_order(tmp_path: Path) -> None:
    rng = np.random.default_rng(3001)
    matrix = rng.normal(size=(8, 12))
    pca = compute_pca(matrix)
    folder = tmp_path / "pca_exports"
    case_ids = [f"case-{index}" for index in range(8)]
    write_pca_exports(
        folder,
        case_ids,
        ["T1", "L1"],
        2,
        pca["mean"],
        pca["modes"],
        pca["eigenvalues"],
        pca["scores"],
    )
    table = pd.read_csv(folder / "scores.csv").iloc[::-1]
    table.to_csv(folder / "scores.csv", index=False)

    with pytest.raises(ValueError, match="case order"):
        pack_pca_exports(folder, tmp_path / "pca.npz", mode_count=4)
