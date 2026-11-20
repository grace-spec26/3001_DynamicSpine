"""Create ShapeWorks workbooks with optional collision-safe staged meshes."""

from __future__ import annotations

import csv
import hashlib
import os
import shutil
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

from openpyxl import load_workbook

from completion_inventory import LEVELS
from pipeline_provenance import validate_materialized


def _digest(path: Path) -> bytes:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.digest()


def _stage_mesh(source: Path, target: Path) -> Path:
    validate_materialized(source)
    target.parent.mkdir(parents=True, exist_ok=True)
    source_digest = _digest(source)
    if target.exists() and target.is_file() and target.stat().st_size > 0:
        if _digest(target) == source_digest:
            return target
    partial = target.with_name(f"{target.name}.partial")
    try:
        shutil.copy2(source, partial)
        if _digest(partial) != source_digest:
            raise ValueError(f"copied mesh differs from source: {partial}")
        os.replace(partial, target)
    finally:
        if partial.exists():
            partial.unlink()
    return target


def create_project(
    package_root: Path,
    output_dir: Path,
    template_path: Path,
    subset: str = "all",
    unique_mesh_names: bool = False,
    output_name: str | None = None,
) -> Path:
    """Write a demo or full ShapeWorks workbook and return its path."""
    if subset not in {"demo", "all"}:
        raise ValueError("subset must be 'demo' or 'all'")
    package_root = Path(package_root)
    output_dir = Path(output_dir)
    template_path = Path(template_path)
    validate_materialized(template_path)
    with (package_root / "data" / "cases.csv").open(
        newline="", encoding="utf-8"
    ) as handle:
        rows = list(csv.DictReader(handle))
    if subset == "demo":
        rows = [row for row in rows if row.get("demo") == "1"]

    output_dir.mkdir(parents=True, exist_ok=True)
    suffix = "_unique" if unique_mesh_names else ""
    if output_name is not None:
        candidate = Path(output_name)
        if candidate.name != output_name or candidate.suffix.lower() != ".xlsx":
            raise ValueError("output_name must be a single .xlsx filename")
        workbook_path = output_dir / candidate
    else:
        workbook_path = output_dir / f"shapeworks_{subset}{suffix}.xlsx"
    workbook = load_workbook(template_path)
    sheet = workbook["data"]
    if sheet.max_row > 1:
        sheet.delete_rows(2, sheet.max_row - 1)

    staged: dict[tuple[str, str], Path] = {}
    if unique_mesh_names:
        jobs = []
        with ThreadPoolExecutor(max_workers=32) as executor:
            for row in rows:
                case_id = row["case_id"]
                for level in LEVELS:
                    source = package_root / "data" / "meshes" / case_id / f"{level}.ply"
                    target = (
                        output_dir
                        / "shapeworks_inputs"
                        / case_id
                        / f"{case_id}_{level}.ply"
                    )
                    jobs.append((case_id, level, executor.submit(_stage_mesh, source, target)))
            for case_id, level, future in jobs:
                staged[(case_id, level)] = future.result()

    for row in rows:
        case_id = row["case_id"]
        values: list[str] = [case_id, ""]
        for level in LEVELS:
            source = package_root / "data" / "meshes" / case_id / f"{level}.ply"
            selected = staged[(case_id, level)] if unique_mesh_names else source
            if not unique_mesh_names:
                validate_materialized(selected)
            values.append(Path(os.path.relpath(selected, output_dir)).as_posix())
        sheet.append(values)

    workbook.save(workbook_path)
    return workbook_path
