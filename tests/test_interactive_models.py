import importlib.util
import sys
from pathlib import Path

import joblib
import numpy as np
import pytest

from student_tasks import fit_pls


SCRIPT_PATH = Path(__file__).resolve().parents[1] / "scripts" / "07_interactive.py"
SPEC = importlib.util.spec_from_file_location("interactive_viewer", SCRIPT_PATH)
assert SPEC and SPEC.loader
viewer = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(viewer)


def test_resolve_viewer_paths_prefers_k3_for_default_gui(
    tmp_path: Path, monkeypatch
) -> None:
    output = tmp_path / "outputs"
    monkeypatch.setattr(viewer, "OUTPUT", output)
    suite = output / "pca_models"
    preferred = suite / "pca_03" / "interactive_model.joblib"
    preferred.parent.mkdir(parents=True)
    preferred.write_bytes(b"model")

    paths = viewer.resolve_viewer_paths(None, None, suite, None, False)

    assert paths.model_path == preferred
    assert paths.suite_dir == suite
    assert paths.output_path == preferred.parent / "interactive_particles.ply"
    assert paths.fixed_output is False


def test_resolve_viewer_paths_preserves_headless_and_missing_k3_fallbacks(
    tmp_path: Path, monkeypatch
) -> None:
    output = tmp_path / "outputs"
    monkeypatch.setattr(viewer, "OUTPUT", output)
    suite = output / "pca_models"

    headless = viewer.resolve_viewer_paths(None, None, suite, None, True)
    gui_without_k3 = viewer.resolve_viewer_paths(None, None, suite, None, False)

    expected = output / "interactive_model.joblib"
    assert headless.model_path == expected
    assert headless.output_path == output / "interactive_particles.ply"
    assert gui_without_k3.model_path == expected


def test_resolve_viewer_paths_honours_all_explicit_paths(tmp_path: Path) -> None:
    run_dir = tmp_path / "run"
    model = tmp_path / "chosen" / "model.joblib"
    suite = tmp_path / "chosen-suite"
    output = tmp_path / "chosen.ply"

    paths = viewer.resolve_viewer_paths(model, run_dir, suite, output, False)

    assert paths.model_path == model
    assert paths.suite_dir == suite
    assert paths.output_path == output
    assert paths.fixed_output is True


def test_resolve_viewer_paths_treats_absolute_output_run_dir_as_explicit(
    tmp_path: Path, monkeypatch
) -> None:
    output = (tmp_path / "outputs").resolve()
    monkeypatch.setattr(viewer, "OUTPUT", output)
    preferred = output / "pca_models/pca_03/interactive_model.joblib"
    preferred.parent.mkdir(parents=True)
    preferred.write_bytes(b"model")

    paths = viewer.resolve_viewer_paths(None, output, None, None, False)

    assert paths.model_path == output / "interactive_model.joblib"


def test_resolve_viewer_paths_keeps_headless_output_in_explicit_run_dir(
    tmp_path: Path,
) -> None:
    run_dir = tmp_path / "exports"
    model = tmp_path / "read-only-models" / "interactive_model.joblib"

    paths = viewer.resolve_viewer_paths(model, run_dir, None, None, True)

    assert paths.model_path == model
    assert paths.output_path == run_dir / "interactive_particles.ply"


def _write_model_bundle(
    folder: Path,
    mode_count: int,
    finite: bool = True,
    finite_estimator: bool = True,
    centre_offset: float = 0.0,
    mean_offset: float = 0.0,
) -> Path:
    folder.mkdir(parents=True, exist_ok=True)
    cohort_rng = np.random.default_rng(3001)
    score_rng = np.random.default_rng(4000 + mode_count)
    case_ids = [f"case-{index}" for index in range(12)]
    centres = cohort_rng.normal(size=(12, 15, 3)) + centre_offset
    features = centres.reshape(12, 45)
    scores = score_rng.normal(size=(12, mode_count))
    model = fit_pls(features, scores, 1)
    if not finite_estimator:
        model.coef_.flat[0] = np.nan
    mean = np.full((3840, 3), mean_offset)
    modes = np.zeros((mode_count, 11520))
    modes[np.arange(mode_count), np.arange(mode_count)] = 1
    if not finite:
        mean[0, 0] = np.nan
    bundle = {
        "model": model,
        "mean": mean,
        "modes": modes,
        "eigenvalues": np.arange(mode_count, 0, -1, dtype=float),
        "case_ids": np.array(case_ids),
        "centres": centres,
        "default_case": case_ids[-1],
        "schema_version": 2,
    }
    model_path = folder / "interactive_model.joblib"
    joblib.dump(bundle, model_path)
    np.savez(
        folder / "pca.npz",
        mean=mean,
        modes=modes,
        eigenvalues=bundle["eigenvalues"],
        case_ids=case_ids,
        scores=scores,
    )
    return model_path


def test_discover_models_includes_complete_finite_models_only(tmp_path: Path) -> None:
    suite = tmp_path / "suite"
    active = _write_model_bundle(suite / "pca_03", 3)
    _write_model_bundle(suite / "pca_04", 4)
    _write_model_bundle(suite / "pca_05", 5, finite=False)
    partial = suite / "pca_06"
    partial.mkdir(parents=True)
    _write_model_bundle(tmp_path / "orphan", 6).replace(
        partial / "interactive_model.joblib"
    )

    model, choices = viewer.discover_models(active, suite)

    assert len(model.modes) == 3
    assert list(choices) == [3, 4]
    assert choices[4].parent.name == "pca_04"


def test_discovery_omits_valid_models_from_a_different_case_or_mean_frame(
    tmp_path: Path,
) -> None:
    suite = tmp_path / "suite"
    active = _write_model_bundle(suite / "pca_03", 3)
    _write_model_bundle(suite / "pca_04", 4, centre_offset=1.0)
    _write_model_bundle(suite / "pca_05", 5, mean_offset=1.0)

    _, choices = viewer.discover_models(active, suite, case="case-0")

    assert list(choices) == [3]


def test_gui_startup_falls_back_when_automatic_k3_is_corrupt(
    tmp_path: Path, monkeypatch
) -> None:
    output = tmp_path / "outputs"
    monkeypatch.setattr(viewer, "OUTPUT", output)
    root_model = _write_model_bundle(output, 6)
    preferred = _write_model_bundle(output / "pca_models/pca_03", 3)
    preferred.write_bytes(b"truncated")
    paths = viewer.resolve_viewer_paths(None, None, None, None, False)

    model, _, active_path = viewer.load_gui_models(paths, case="case-0")

    assert len(model.modes) == 6
    assert active_path == root_model.resolve()


def test_discovery_requires_active_companion_pca(tmp_path: Path) -> None:
    suite = tmp_path / "suite"
    active = _write_model_bundle(suite / "pca_03", 3)
    (active.parent / "pca.npz").unlink()

    with pytest.raises(ValueError, match="Inconsistent model bundle"):
        viewer.discover_models(active, suite)


def test_discovery_ignores_noncanonical_suite_folders(tmp_path: Path) -> None:
    suite = tmp_path / "suite"
    active = _write_model_bundle(suite / "pca_03", 3)
    _write_model_bundle(suite / "pca_backup", 6)

    _, choices = viewer.discover_models(active, suite)

    assert list(choices) == [3]


def test_student_spine_model_switches_pca_mode_count_via_bundle(tmp_path: Path) -> None:
    model_path = _write_model_bundle(tmp_path / "pca_04", 4)
    model = viewer.StudentSpineModel(model_path)
    keypoints = model.centres[0]

    scores, particles = model.predict(keypoints)

    assert scores.shape == (4,)
    assert particles.shape == (3840, 3)
    assert len(model.split(particles)) == 15


def test_interactive_session_retains_edits_across_switch_and_reset(
    tmp_path: Path,
) -> None:
    suite = tmp_path / "suite"
    active_path = _write_model_bundle(suite / "pca_03", 3)
    target_path = _write_model_bundle(suite / "pca_06", 6)
    model = viewer.StudentSpineModel(active_path, active_path.parent / "pca.npz")
    session = viewer.InteractiveSession(
        model,
        active_path,
        "case-0",
        active_path.parent / "interactive_particles.ply",
        fixed_output=False,
    )
    original = session.keypoints.copy()

    session.edit_point(0, 0, float(original[0, 0] + 2.5))
    edited = session.keypoints.copy()
    scores, particles = session.switch_model(target_path)

    np.testing.assert_allclose(session.keypoints, edited)
    assert len(session.model.modes) == 6
    assert scores.shape == (6,)
    assert particles.shape == (3840, 3)
    assert session.output_path == target_path.parent / "interactive_particles.ply"
    session.reset()
    np.testing.assert_allclose(session.keypoints, original)


def test_interactive_session_failed_switch_is_atomic_and_fixed_output_stays_fixed(
    tmp_path: Path,
) -> None:
    suite = tmp_path / "suite"
    active_path = _write_model_bundle(suite / "pca_03", 3)
    incompatible = _write_model_bundle(
        suite / "pca_06", 6, centre_offset=1.0
    )
    fixed_output = tmp_path / "chosen.ply"
    model = viewer.StudentSpineModel(active_path, active_path.parent / "pca.npz")
    session = viewer.InteractiveSession(
        model, active_path, "case-0", fixed_output, fixed_output=True
    )
    session.edit_point(2, 1, float(session.keypoints[2, 1] - 1.25))
    before_keypoints = session.keypoints.copy()
    before_scores, before_particles = session.predict()

    with pytest.raises(ValueError, match="different input landmarks"):
        session.switch_model(incompatible)

    assert session.model_path == active_path.resolve()
    assert session.output_path == fixed_output
    np.testing.assert_allclose(session.keypoints, before_keypoints)
    after_scores, after_particles = session.predict()
    np.testing.assert_allclose(after_scores, before_scores)
    np.testing.assert_allclose(after_particles, before_particles)


def test_interactive_session_save_writes_every_predicted_particle(
    tmp_path: Path,
) -> None:
    model_path = _write_model_bundle(tmp_path / "pca_03", 3)
    model = viewer.StudentSpineModel(model_path, model_path.parent / "pca.npz")
    output = tmp_path / "saved.ply"
    session = viewer.InteractiveSession(
        model, model_path, "case-0", output, fixed_output=True
    )
    _, expected = session.predict()

    saved = session.save()
    actual = np.loadtxt(saved, skiprows=7)

    assert actual.shape == (3840, 3)
    assert np.isfinite(actual).all()
    np.testing.assert_allclose(actual, expected, rtol=1e-10, atol=1e-10)


def test_interactive_session_switch_requires_companion_and_expected_mode_count(
    tmp_path: Path,
) -> None:
    suite = tmp_path / "suite"
    active_path = _write_model_bundle(suite / "pca_03", 3)
    target_path = _write_model_bundle(suite / "pca_06", 6)
    model = viewer.StudentSpineModel(active_path, active_path.parent / "pca.npz")
    session = viewer.InteractiveSession(
        model, active_path, "case-0", tmp_path / "prediction.ply", True
    )

    with pytest.raises(ValueError, match="expected 4 PCA modes"):
        session.switch_model(target_path, expected_mode_count=4)
    (target_path.parent / "pca.npz").unlink()
    with pytest.raises(ValueError, match="Inconsistent model bundle"):
        session.switch_model(target_path, expected_mode_count=6)
    assert session.model_path == active_path.resolve()


def test_failed_switch_returns_retained_prediction_for_pending_edit(
    tmp_path: Path,
) -> None:
    suite = tmp_path / "suite"
    active_path = _write_model_bundle(suite / "pca_03", 3)
    incompatible = _write_model_bundle(
        suite / "pca_06", 6, centre_offset=1.0
    )
    model = viewer.StudentSpineModel(active_path, active_path.parent / "pca.npz")
    session = viewer.InteractiveSession(
        model, active_path, "case-0", tmp_path / "prediction.ply", True
    )
    original_particles = session.predict()[1]
    session.edit_point(0, 0, float(session.keypoints[0, 0] + 3.0))

    switched, prediction, error = viewer.attempt_session_switch(
        session, incompatible, expected_mode_count=6
    )

    assert switched is False
    assert "different input landmarks" in error
    np.testing.assert_allclose(prediction[1], session.predict()[1])
    assert not np.allclose(prediction[1], original_particles)


def test_headless_main_rejects_model_without_companion_pca(tmp_path: Path) -> None:
    model_path = _write_model_bundle(tmp_path / "pca_03", 3)
    (model_path.parent / "pca.npz").unlink()

    with pytest.raises(SystemExit) as exit_info:
        viewer.main(["--model", str(model_path), "--headless"])

    assert exit_info.value.code == 1


def test_invalid_nonfinite_model_bundle_is_rejected(tmp_path: Path) -> None:
    model_path = _write_model_bundle(tmp_path / "pca_03", 3, finite=False)

    try:
        viewer.StudentSpineModel(model_path)
    except ValueError as error:
        assert "Inconsistent model bundle" in str(error)
    else:
        raise AssertionError("non-finite model bundle was accepted")


def test_headless_main_exports_ply_without_importing_gui_dependencies(
    tmp_path: Path,
) -> None:
    model_path = _write_model_bundle(tmp_path / "pca_03", 3)
    output = tmp_path / "prediction.ply"
    sys.modules.pop("PyQt5", None)

    result = viewer.main(
        [
            "--model",
            str(model_path),
            "--case",
            "case-0",
            "--headless",
            "--output",
            str(output),
        ]
    )

    assert result == 0
    assert "PyQt5" not in sys.modules
    text = output.read_text(encoding="ascii")
    assert text.startswith("ply\nformat ascii 1.0\nelement vertex 3840\n")
    assert "end_header\n" in text
    actual = np.loadtxt(output, skiprows=7)
    model = viewer.StudentSpineModel(model_path, model_path.parent / "pca.npz")
    _, expected = model.predict(model.centres[model.case_ids.index("case-0")])
    assert actual.shape == (3840, 3)
    assert np.isfinite(actual).all()
    np.testing.assert_allclose(actual, expected, rtol=1e-9, atol=1e-9)


def test_discovery_skips_nonfinite_estimator_and_truncated_optional_bundle(
    tmp_path: Path,
) -> None:
    suite = tmp_path / "suite"
    active = _write_model_bundle(suite / "pca_03", 3)
    _write_model_bundle(suite / "pca_04", 4, finite_estimator=False)
    broken = suite / "pca_05"
    broken.mkdir(parents=True)
    (broken / "interactive_model.joblib").write_bytes(b"truncated")
    np.savez(broken / "pca.npz", modes=np.zeros((5, 11520)))

    _, choices = viewer.discover_models(active, suite)

    assert list(choices) == [3]


def test_discovery_rejects_joblib_and_pca_bundle_mismatch(tmp_path: Path) -> None:
    suite = tmp_path / "suite"
    active = _write_model_bundle(suite / "pca_03", 3)
    candidate = _write_model_bundle(suite / "pca_04", 4)
    pca_path = candidate.parent / "pca.npz"
    bundle = dict(np.load(pca_path))
    bundle["modes"] = bundle["modes"].copy()
    bundle["modes"][0, 0] += 1
    np.savez(pca_path, **bundle)

    _, choices = viewer.discover_models(active, suite)

    assert list(choices) == [3]
