import csv
import hashlib
import json
from pathlib import Path

import pytest
from openpyxl import Workbook, load_workbook

from completion_inventory import (
    LEVELS,
    build_inventory,
    validate_cohort,
    verify_sha256_manifest,
)
from create_project import create_project


def _make_package(root: Path, case_ids: tuple[str, ...] = ("case-b", "case-a")) -> None:
    data = root / "data"
    (data / "masks").mkdir(parents=True)
    (data / "transforms").mkdir()
    (data / "merged_spines").mkdir()
    (data / "meshes").mkdir()
    with (data / "cases.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=(
                "case_id",
                "mask",
                "transform",
                "merged_mesh",
                "regression_split",
                "demo",
            ),
        )
        writer.writeheader()
        for case_id in case_ids:
            writer.writerow(
                {
                    "case_id": case_id,
                    "mask": f"data/masks/{case_id}.nii.gz",
                    "transform": f"data/transforms/{case_id}.json",
                    "merged_mesh": f"data/merged_spines/{case_id}.ply",
                    "regression_split": "train",
                    "demo": "1",
                }
            )
            (data / "masks" / f"{case_id}.nii.gz").write_bytes(b"nifti")
            (data / "transforms" / f"{case_id}.json").write_text(
                json.dumps({"world_ras_to_aligned": [
                    [1, 0, 0, 0],
                    [0, 1, 0, 0],
                    [0, 0, 1, 0],
                    [0, 0, 0, 1],
                ]}),
                encoding="utf-8",
            )
            (data / "merged_spines" / f"{case_id}.ply").write_text(
                "ply\n", encoding="utf-8"
            )
            case_meshes = data / "meshes" / case_id
            case_meshes.mkdir()
            for level in LEVELS:
                (case_meshes / f"{level}.ply").write_text(
                    f"ply\n{case_id}-{level}\n", encoding="utf-8"
                )


def _make_template(path: Path) -> None:
    workbook = Workbook()
    sheet = workbook.active
    sheet.title = "data"
    sheet.append(["name", "notes", *LEVELS])
    sheet.append(["example", "", *["mesh.ply"] * len(LEVELS)])
    workbook.save(path)


def test_inventory_preserves_case_and_anatomical_domain_order(tmp_path: Path) -> None:
    _make_package(tmp_path)

    inventory = build_inventory(tmp_path)
    validate_cohort(inventory, expected_cases=2, expected_domains=15)

    assert inventory["case_ids"] == ["case-b", "case-a"]
    assert inventory["domains"] == list(LEVELS)
    assert inventory["mesh_count"] == 30
    assert sorted(inventory["duplicate_basenames"]) == sorted(
        f"{level}.ply" for level in LEVELS
    )


def test_inventory_rejects_missing_domain(tmp_path: Path) -> None:
    _make_package(tmp_path, ("case-a",))
    (tmp_path / "data/meshes/case-a/T12.ply").unlink()

    inventory = build_inventory(tmp_path)

    with pytest.raises(ValueError, match="missing"):
        validate_cohort(inventory, expected_cases=1, expected_domains=15)


def test_inventory_rejects_empty_placeholder(tmp_path: Path) -> None:
    _make_package(tmp_path, ("case-a",))
    (tmp_path / "data/meshes/case-a/T1.ply").write_bytes(b"")

    inventory = build_inventory(tmp_path)

    with pytest.raises(ValueError, match="invalid"):
        validate_cohort(inventory, expected_cases=1, expected_domains=15)


def test_verify_sha256_manifest_detects_changed_file(tmp_path: Path) -> None:
    payload = tmp_path / "payload.bin"
    payload.write_bytes(b"original")
    checksums = tmp_path / "SHA256SUMS.txt"
    checksums.write_text(
        f"{hashlib.sha256(b'original').hexdigest()}  payload.bin\n", encoding="utf-8"
    )
    verify_sha256_manifest(tmp_path, checksums)

    payload.write_bytes(b"changed")

    with pytest.raises(ValueError, match="checksum"):
        verify_sha256_manifest(tmp_path, checksums)


def test_unique_project_stages_collision_safe_meshes_without_changing_raw_data(
    tmp_path: Path,
) -> None:
    _make_package(tmp_path)
    template = tmp_path / "template.xlsx"
    _make_template(template)
    output = tmp_path / "outputs"
    before = {
        path.relative_to(tmp_path): hashlib.sha256(path.read_bytes()).hexdigest()
        for path in (tmp_path / "data").rglob("*")
        if path.is_file()
    }

    workbook_path = create_project(
        package_root=tmp_path,
        output_dir=output,
        template_path=template,
        subset="all",
        unique_mesh_names=True,
    )

    after = {
        path.relative_to(tmp_path): hashlib.sha256(path.read_bytes()).hexdigest()
        for path in (tmp_path / "data").rglob("*")
        if path.is_file()
    }
    assert after == before
    staged = sorted((output / "shapeworks_inputs").rglob("*.ply"))
    assert len(staged) == 30
    assert len({path.name for path in staged}) == 30
    for path in staged:
        case_id, level = path.stem.rsplit("_", 1)
        source = tmp_path / "data/meshes" / case_id / f"{level}.ply"
        assert hashlib.sha256(path.read_bytes()).digest() == hashlib.sha256(
            source.read_bytes()
        ).digest()

    sheet = load_workbook(workbook_path, read_only=True, data_only=True)["data"]
    rows = list(sheet.iter_rows(min_row=2, values_only=True))
    assert [row[0] for row in rows] == ["case-b", "case-a"]
    assert all("shapeworks_inputs" in value for row in rows for value in row[2:])


def test_unique_project_repairs_an_interrupted_partial_copy(tmp_path: Path) -> None:
    _make_package(tmp_path, ("case-a",))
    template = tmp_path / "template.xlsx"
    _make_template(template)
    output = tmp_path / "outputs"
    partial_target = output / "shapeworks_inputs/case-a/case-a_T1.ply"
    partial_target.parent.mkdir(parents=True)
    partial_target.write_bytes(b"partial")

    create_project(
        package_root=tmp_path,
        output_dir=output,
        template_path=template,
        subset="all",
        unique_mesh_names=True,
    )

    source = tmp_path / "data/meshes/case-a/T1.ply"
    assert partial_target.read_bytes() == source.read_bytes()
    assert not list(partial_target.parent.glob("*.partial"))


def test_project_can_publish_the_explicit_full_run_workbook(tmp_path: Path) -> None:
    _make_package(tmp_path, ("case-a",))
    template = tmp_path / "template.xlsx"
    _make_template(template)

    workbook = create_project(
        package_root=tmp_path,
        output_dir=tmp_path / "outputs",
        template_path=template,
        subset="all",
        unique_mesh_names=True,
        output_name="full_run.xlsx",
    )

    assert workbook.name == "full_run.xlsx"
    sheet = load_workbook(workbook, read_only=True, data_only=True)["data"]
    assert sheet.max_row == 2
    assert sheet.max_column == 17


def test_project_rejects_output_name_outside_output_directory(tmp_path: Path) -> None:
    _make_package(tmp_path, ("case-a",))
    template = tmp_path / "template.xlsx"
    _make_template(template)

    with pytest.raises(ValueError, match="output_name"):
        create_project(
            package_root=tmp_path,
            output_dir=tmp_path / "outputs",
            template_path=template,
            output_name="../outside.xlsx",
        )
