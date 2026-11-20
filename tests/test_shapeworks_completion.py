from pathlib import Path

import numpy as np
import pytest

from shapeworks_completion import (
    build_particle_store,
    run_shapeworks,
    run_shapeworks_with_retry,
    stage_reference_particles,
    validate_particle_order,
    validate_particle_cache,
    validate_particle_store,
    validate_particle_set,
    write_particle_validation_cache,
)


def _write_particles(
    root: Path,
    case_ids: tuple[str, ...] = ("case-a", "case-b"),
    domains: tuple[str, ...] = ("T1", "L1"),
    rows: int = 3,
) -> None:
    root.mkdir(parents=True, exist_ok=True)
    for case_index, case_id in enumerate(case_ids):
        for domain_index, domain in enumerate(domains):
            values = np.arange(rows * 3, dtype=float).reshape(rows, 3)
            values += case_index * 100 + domain_index * 10
            for coordinate_system in ("local", "world"):
                np.savetxt(
                    root / f"{case_id}_{domain}_{coordinate_system}.particles", values
                )


def test_validate_particle_set_preserves_case_domain_and_coordinate_order(
    tmp_path: Path,
) -> None:
    particles = tmp_path / "particles"
    _write_particles(particles)

    result = validate_particle_set(
        particles, ["case-b", "case-a"], ["L1", "T1"], particles_per_domain=3
    )

    assert result["case_ids"] == ["case-b", "case-a"]
    assert result["domains"] == ["L1", "T1"]
    assert result["file_count"] == 8
    assert [Path(value).name for value in result["paths"][:4]] == [
        "case-b_L1_local.particles",
        "case-b_L1_world.particles",
        "case-b_T1_local.particles",
        "case-b_T1_world.particles",
    ]


@pytest.mark.parametrize("failure", ["missing", "wrong_rows", "nonfinite", "extra"])
def test_validate_particle_set_rejects_incomplete_or_malformed_sets(
    tmp_path: Path, failure: str
) -> None:
    particles = tmp_path / "particles"
    _write_particles(particles, ("case-a",), ("T1",), rows=3)
    local = particles / "case-a_T1_local.particles"
    if failure == "missing":
        local.unlink()
    elif failure == "wrong_rows":
        np.savetxt(local, np.zeros((2, 3)))
    elif failure == "nonfinite":
        values = np.zeros((3, 3))
        values[1, 2] = np.nan
        np.savetxt(local, values)
    else:
        np.savetxt(particles / "case-x_T1_local.particles", np.zeros((3, 3)))

    with pytest.raises(ValueError):
        validate_particle_set(
            particles, ["case-a"], ["T1"], particles_per_domain=3
        )


def test_stage_reference_particles_copies_only_a_matching_cohort(
    tmp_path: Path,
) -> None:
    reference = tmp_path / "reference"
    destination = tmp_path / "destination"
    _write_particles(reference)
    destination.mkdir()
    (destination / "keep.txt").write_text("unrelated", encoding="utf-8")

    copied = stage_reference_particles(
        reference,
        destination,
        ["case-a", "case-b"],
        ["T1", "L1"],
        particles_per_domain=3,
    )

    assert len(copied) == 8
    assert (destination / "keep.txt").read_text(encoding="utf-8") == "unrelated"
    validate_particle_set(
        destination,
        ["case-a", "case-b"],
        ["T1", "L1"],
        particles_per_domain=3,
        allow_unrelated=True,
    )


def test_stage_reference_particles_rejects_mismatched_cohort(tmp_path: Path) -> None:
    reference = tmp_path / "reference"
    _write_particles(reference, ("case-a",), ("T1",), rows=3)

    with pytest.raises(ValueError, match="missing"):
        stage_reference_particles(
            reference,
            tmp_path / "destination",
            ["case-b"],
            ["T1"],
            particles_per_domain=3,
        )


def _make_fake_shapeworks(path: Path, fail_first: bool) -> None:
    path.write_text(
        "#!/usr/bin/env python3\n"
        "import pathlib, sys\n"
        "state = pathlib.Path(__file__).with_suffix('.state')\n"
        "count = int(state.read_text()) if state.exists() else 0\n"
        "state.write_text(str(count + 1))\n"
        "print('fake shapeworks', ' '.join(sys.argv[1:]))\n"
        f"raise SystemExit(1 if {fail_first!r} and count == 0 else 0)\n",
        encoding="utf-8",
    )
    path.chmod(0o755)


def test_run_shapeworks_writes_command_output_to_log(tmp_path: Path) -> None:
    executable = tmp_path / "shapeworks"
    _make_fake_shapeworks(executable, fail_first=False)
    project = tmp_path / "project.xlsx"
    project.write_bytes(b"workbook")
    log = tmp_path / "run.log"

    result = run_shapeworks(project, executable, log)

    assert result.returncode == 0
    assert "fake shapeworks optimize --name" in log.read_text(encoding="utf-8")
    assert str(project) in log.read_text(encoding="utf-8")


def test_run_shapeworks_with_retry_makes_exactly_two_attempts(tmp_path: Path) -> None:
    executable = tmp_path / "shapeworks"
    _make_fake_shapeworks(executable, fail_first=True)
    project = tmp_path / "project.xlsx"
    project.write_bytes(b"workbook")

    results = run_shapeworks_with_retry(
        project, executable, tmp_path / "logs", attempts=2
    )

    assert [result.returncode for result in results] == [1, 0]
    assert executable.with_suffix(".state").read_text(encoding="utf-8") == "2"
    assert sorted(path.name for path in (tmp_path / "logs").glob("*.log")) == [
        "shapeworks_attempt_1.log",
        "shapeworks_attempt_2.log",
    ]


def test_zero_exit_with_invalid_artifacts_is_retried(tmp_path: Path) -> None:
    executable = tmp_path / "shapeworks"
    _make_fake_shapeworks(executable, fail_first=False)
    project = tmp_path / "project.xlsx"
    project.write_bytes(b"workbook")
    validations = {"count": 0}

    def validate_attempt() -> None:
        validations["count"] += 1
        raise ValueError("missing particles")

    results = run_shapeworks_with_retry(
        project,
        executable,
        tmp_path / "logs",
        attempts=2,
        validate_attempt=validate_attempt,
    )

    assert len(results) == 2
    assert [result.returncode for result in results] == [0, 0]
    assert [result.valid for result in results] == [False, False]
    assert validations["count"] == 2


def test_launch_failures_are_logged_and_retried(tmp_path: Path) -> None:
    project = tmp_path / "project.xlsx"
    project.write_bytes(b"workbook")

    results = run_shapeworks_with_retry(
        project,
        tmp_path / "missing-shapeworks",
        tmp_path / "logs",
        attempts=2,
    )

    assert len(results) == 2
    assert all(result.returncode != 0 and not result.valid for result in results)
    assert "not runnable" in (tmp_path / "logs/shapeworks_attempt_1.log").read_text()


def test_particle_validation_cache_checks_published_inventory_and_stable_source(
    tmp_path: Path,
) -> None:
    reference = tmp_path / "reference"
    destination = tmp_path / "destination"
    _write_particles(reference, ("case-a",), ("T1",), rows=3)
    stage_reference_particles(
        reference,
        destination,
        ["case-a"],
        ["T1"],
        particles_per_domain=3,
    )
    cache = tmp_path / "particles.validation.json"
    write_particle_validation_cache(
        reference,
        cache,
        ["case-a"],
        ["T1"],
        particles_per_domain=3,
    )

    result = validate_particle_cache(
        destination,
        cache,
        ["case-a"],
        ["T1"],
        particles_per_domain=3,
        content_root=reference,
    )

    assert result["file_count"] == 2
    (destination / "case-a_T1_local.particles").write_text("short\n", encoding="utf-8")
    with pytest.raises(ValueError, match="size"):
        validate_particle_cache(
            destination,
            cache,
            ["case-a"],
            ["T1"],
            particles_per_domain=3,
            content_root=reference,
        )


def test_reference_order_metadata_rejects_same_files_in_different_order(
    tmp_path: Path,
) -> None:
    order = tmp_path / "particle_order.json"
    order.write_text(
        '{"case_ids":["case-b","case-a"],"domains":["T1","L1"]}\n',
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="case order"):
        validate_particle_order(order, ["case-a", "case-b"], ["T1", "L1"])


def test_particle_store_is_the_ordered_finite_analysis_source(tmp_path: Path) -> None:
    particles = tmp_path / "particles"
    _write_particles(particles, ("case-a", "case-b"), ("T1", "L1"), rows=3)
    store = build_particle_store(
        particles,
        tmp_path / "particle_store.npz",
        ["case-a", "case-b"],
        ["T1", "L1"],
        particles_per_domain=3,
    )

    result = validate_particle_store(
        store,
        ["case-a", "case-b"],
        ["T1", "L1"],
        particles_per_domain=3,
    )

    assert result["world"].shape == (2, 2, 3, 3)
    assert result["local"].shape == (2, 2, 3, 3)
    with pytest.raises(ValueError, match="case order"):
        validate_particle_store(
            store,
            ["case-b", "case-a"],
            ["T1", "L1"],
            particles_per_domain=3,
        )
