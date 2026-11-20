import json
from pathlib import Path

import pytest

from complete_pipeline import (
    PipelineContext,
    Stage,
    completion_exit_code,
    run_stages,
    summarize_sources,
    write_completion_report,
)


def _context(tmp_path: Path, allow_fallback: bool = False) -> PipelineContext:
    return PipelineContext(
        workshop=tmp_path / "workshop",
        reference=tmp_path / "reference",
        allow_reference_fallback=allow_fallback,
    )


def test_run_stages_preserves_declared_order(tmp_path: Path) -> None:
    events: list[str] = []
    completed: set[str] = set()

    def stage(name: str) -> Stage:
        def action(context: PipelineContext) -> None:
            events.append(name)
            completed.add(name)

        def validate(context: PipelineContext) -> None:
            if name not in completed:
                raise ValueError(f"{name} incomplete")

        return Stage(name, action, validate)

    result = run_stages(
        [stage("particles"), stage("pca"), stage("models")],
        _context(tmp_path),
        tmp_path / "state.json",
        resume=False,
    )

    assert events == ["particles", "pca", "models"]
    assert result["completed"] == ["particles", "pca", "models"]


def test_resume_validates_and_skips_an_existing_stage(tmp_path: Path) -> None:
    state_path = tmp_path / "state.json"
    state_path.write_text(json.dumps({"completed": ["particles"]}), encoding="utf-8")
    called: list[str] = []
    stage = Stage(
        "particles",
        lambda context: called.append("action"),
        lambda context: called.append("validate"),
    )

    result = run_stages(
        [stage], _context(tmp_path), state_path, resume=True
    )

    assert called == ["validate"]
    assert result["skipped"] == ["particles"]


def test_validation_failure_stops_before_later_stage(tmp_path: Path) -> None:
    events: list[str] = []

    failing = Stage(
        "pca",
        lambda context: events.append("pca-action"),
        lambda context: (_ for _ in ()).throw(ValueError("bad PCA")),
    )
    later = Stage(
        "models",
        lambda context: events.append("models-action"),
        lambda context: None,
    )

    with pytest.raises(ValueError, match="bad PCA"):
        run_stages(
            [failing, later], _context(tmp_path), tmp_path / "state.json"
        )

    assert events == ["pca-action"]


def test_fallback_permission_reaches_stage_action(tmp_path: Path) -> None:
    observed: list[bool] = []
    completed = {"value": False}

    def action(context: PipelineContext) -> None:
        observed.append(context.allow_reference_fallback)
        completed["value"] = True

    def validate(context: PipelineContext) -> None:
        if not completed["value"]:
            raise ValueError("not complete")

    run_stages(
        [Stage("particles", action, validate)],
        _context(tmp_path, allow_fallback=True),
        tmp_path / "state.json",
    )

    assert observed == [True]


def test_source_summary_counts_each_artifact_family() -> None:
    manifest = {
        "artifacts": {
            "centroids": {"source": "workshop_reused"},
            "particles": {"source": "reference_fallback"},
            "pca": {"source": "workshop_regenerated"},
            "models": {"source": "workshop_regenerated"},
        }
    }

    assert summarize_sources(manifest) == {
        "workshop_regenerated": 2,
        "workshop_reused": 1,
        "reference_fallback": 1,
    }


def test_incomplete_model_suite_returns_nonzero_completion_status(tmp_path: Path) -> None:
    workshop = tmp_path / "workshop"
    (workshop / "outputs/pca_models/pca_03").mkdir(parents=True)

    assert completion_exit_code(workshop) == 1


def test_rebuilt_upstream_forces_valid_downstream_to_rerun(tmp_path: Path) -> None:
    events: list[str] = []
    upstream_ready = {"value": False}

    def upstream_validate(context: PipelineContext) -> None:
        if not upstream_ready["value"]:
            raise ValueError("stale upstream")

    def upstream_action(context: PipelineContext) -> None:
        events.append("upstream")
        upstream_ready["value"] = True

    downstream = Stage(
        "downstream",
        lambda context: events.append("downstream"),
        lambda context: None,
    )

    run_stages(
        [Stage("upstream", upstream_action, upstream_validate), downstream],
        _context(tmp_path),
        tmp_path / "state.json",
        resume=True,
    )

    assert events == ["upstream", "downstream"]


def test_completion_report_uses_observed_manifest_not_historical_constants(
    tmp_path: Path,
) -> None:
    context = _context(tmp_path, allow_fallback=True)
    output = context.workshop / "outputs"
    output.mkdir(parents=True)
    (output / "provenance_manifest.json").write_text(
        json.dumps(
            {
                "failures": ["custom optimizer failure"],
                "artifacts": {
                    "full_run_particles": {
                        "source": "reference_fallback",
                        "reference_commit": "abc123",
                    },
                    "pca": {"source": "workshop_regenerated"},
                },
            }
        ),
        encoding="utf-8",
    )

    report = write_completion_report(
        context, {"completed": ["particles", "pca"], "skipped": []}
    ).read_text(encoding="utf-8")

    assert "abc123" in report
    assert "custom optimizer failure" in report
    assert "830a10d" not in report
