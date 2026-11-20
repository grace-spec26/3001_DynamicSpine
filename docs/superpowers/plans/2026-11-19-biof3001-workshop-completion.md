# BIOF3001 Workshop Completion Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Complete the Workshop's ShapeWorks, PCA, PLS/reconstruction, and interactive-model pipeline from its own 71-case raw dataset, using provenance-labeled reference artifacts only when a locally regenerated stage objectively fails.

**Architecture:** A resumable completion runner coordinates focused validation/provenance helpers and the existing numbered course scripts. Each stage validates case/domain ordering, dimensions, materialization, and numeric finiteness before publishing outputs; fallback occurs at the earliest failed artifact family so downstream stages can still be regenerated locally.

**Tech Stack:** Python 3 in Conda environment `BIOF3001`, NumPy, pandas, SciPy/scikit-learn, nibabel, PyVista/VTK, openpyxl, joblib, pytest, ShapeWorks under `/Applications/ShapeWorks`.

**Spec:** `docs/superpowers/specs/2026-11-19-biof3001-workshop-completion-design.md`

## Global Constraints

- Never modify the Workshop's raw `data/` files.
- Use `conda run -n BIOF3001` for course Python commands; use the vendor runtime only for ShapeWorks operations that require it.
- Use 71 subjects, 15 ordered domains, and 256 particles per domain (3,840 concatenated particles per subject).
- Do not add Procrustes, scale, or pose normalization beyond the supplied prealignment.
- Permit reference fallback only after a reproducible failure and one controlled retry; label every fallback artifact.
- Never treat an iCloud dataless placeholder, zero-byte file, malformed file, or non-finite array as a valid artifact.
- Do not commit virtual environments, caches, `.DS_Store`, model weights, generated datasets, uploads, or outputs.
- The target has no Git metadata or remote at plan time. Do not initialize or claim a commit/push; if a repository is supplied later, follow the user's Git workflow.

## Review Focus

- iCloud placeholders that exist by path but cannot be read must fail materialization validation in Task 1.
- Reference fallback with mismatched case IDs, domain order, or particle counts must be rejected in Task 4.
- Valid PLS component counts larger than the number of response columns must remain accepted in Task 2.
- Mixed path separators in ShapeWorks workbooks must normalize to the correct output-relative path in Task 5.
- A partial model suite or model with non-finite arrays must be excluded from viewer discovery in Task 7.

---

### Task 1: Recovery Snapshot, Inventory, and Provenance Foundation

**Files:**
- Create: `.gitignore`
- Create: `scripts/pipeline_provenance.py`
- Create: `tests/test_pipeline_provenance.py`
- Create at run time: `outputs/provenance_manifest.json`
- Create at run time: `outputs/logs/`
- Create at run time: `_backups/<timestamp>/`

**Interfaces:**
- Produces: `inspect_file(path: Path) -> dict`, `validate_materialized(path: Path) -> None`, `record_artifact(manifest: dict, family: str, source: str, paths: list[Path], metadata: dict) -> None`, and `write_manifest(path: Path, manifest: dict) -> None`.
- Source values are exactly `workshop_regenerated`, `workshop_reused`, or `reference_fallback`.

- [ ] **Step 1: Write failing provenance/materialization tests**

Add tests asserting that readable files report size/hash, empty or dataless/unreadable files raise a clear error, invalid source labels are rejected, and a manifest round-trips with commands, versions, timestamps, checksums, and failure reasons.

- [ ] **Step 2: Run the focused tests and verify failure**

Run: `conda run -n BIOF3001 pytest -q tests/test_pipeline_provenance.py`

Expected: FAIL because `pipeline_provenance` is absent.

- [ ] **Step 3: Implement the provenance helper and ignore rules**

Implement the interfaces above. Ignore `venv/`, `.venv/`, `env/`, `.env/`, `__pycache__/`, `.DS_Store`, `*.pt`, `*.pth`, `*.ckpt`, generated output directories, uploads, and backup directories.

- [ ] **Step 4: Run tests and create the recovery snapshot**

Run the focused tests, inventory the Workshop, and preserve source/configuration plus the existing partial output tree with a same-volume atomic move or APFS clone. Verify `data/` paths and hashes/materialization flags are unchanged.

- [ ] **Step 5: Record the checkpoint**

Run `git status --short` only if the target becomes a Git repository; otherwise record “not a Git repository; no commit or push” in the completion log.

### Task 2: Repair and Verify Student Algorithms

**Files:**
- Modify: `scripts/student_tasks.py`
- Modify: `scripts/check_exercises.py`
- Create: `tests/test_student_tasks.py`

**Interfaces:**
- Produces: `fit_pls(x_train, y_train, n_components)`, `predict_scores(model, features)`, and `reconstruct_particles(scores, mean, modes)` with explicit two-dimensional/finiteness/shape validation.
- Preserves: `label_centroid_ijk`, `apply_affine`, `mesh_row`, and `landmark_features` behavior.

- [ ] **Step 1: Write failing algorithm tests**

Test one-feature and 45-feature PLS inputs, course component counts `(1, 2, 3, 5, 8)` where sample/rank limits permit them, invalid row counts, non-finite values, incompatible reconstruction dimensions, and the existing centroid/affine/mesh/landmark examples.

- [ ] **Step 2: Verify current failures**

Run: `conda run -n BIOF3001 pytest -q tests/test_student_tasks.py && conda run -n BIOF3001 python scripts/check_exercises.py`

Expected: PLS tests/checks fail on the current `(N, 45)` restriction and response-column component cap.

- [ ] **Step 3: Make the minimal validation fixes**

Retain the Workshop algorithm bodies, remove unreachable placeholders, accept any nonempty two-dimensional feature matrix, cap PLS components according to the estimator/sample constraints used by the course, and add reconstruction shape/finiteness validation.

- [ ] **Step 4: Verify algorithms**

Run the focused tests and `scripts/check_exercises.py`; expected result is all checks passing.

- [ ] **Step 5: Record the checkpoint**

If Git is available, stage only code/tests/docs, verify no generated artifacts are staged, commit, and push; otherwise log that no repository exists.

### Task 3: Complete Project Generation and Input Validation

**Files:**
- Modify: `scripts/02_create_project.py`
- Modify: `scripts/00_check_package.py`
- Create: `scripts/completion_inventory.py`
- Create: `tests/test_project_generation.py`

**Interfaces:**
- Produces: `build_inventory(package_root: Path) -> dict`, `validate_cohort(inventory: dict, expected_cases: int = 71, expected_domains: int = 15) -> None`, and the CLI option `02_create_project.py --unique-mesh-names`.
- Consumes: Task 1 materialization/provenance helpers and existing `student_tasks.mesh_row`.

- [ ] **Step 1: Write failing inventory/project tests**

Test exact 71-case/15-domain ordering, duplicate basename detection, unique staged names, SHA verification, a missing domain, an unreadable placeholder, and a raw-data immutability snapshot.

- [ ] **Step 2: Verify current failures**

Run: `conda run -n BIOF3001 pytest -q tests/test_project_generation.py`

Expected: FAIL because inventory helpers and `--unique-mesh-names` are absent.

- [ ] **Step 3: Implement inventory and project parity**

Add unique mesh staging and checksums compatible with the reference project, keeping all writes outside `data/`. Make package checking reject empty/unreadable/dimensionally incomplete inputs.

- [ ] **Step 4: Verify and generate projects**

Run the focused tests, then create/validate demo and full-cohort workbooks in `outputs/`. Confirm 1,065 unique full inputs, 71 centroids, and unchanged raw-data hashes.

- [ ] **Step 5: Record the checkpoint**

Apply the Git/no-Git checkpoint rule from Task 2.

### Task 4: ShapeWorks Regeneration with Validated Fallback

**Files:**
- Create: `scripts/shapeworks_completion.py`
- Create: `tests/test_shapeworks_completion.py`
- Create at run time: `outputs/full_run.xlsx`
- Create at run time: `outputs/full_run_particles/`
- Create at run time: `outputs/shapeworks_inputs/`
- Create at run time: `outputs/logs/shapeworks_*.log`

**Interfaces:**
- Produces: `validate_particle_set(root: Path, case_ids: list[str], domains: list[str], particles_per_domain: int = 256) -> dict`, `run_shapeworks(project: Path, executable: Path, log: Path) -> subprocess.CompletedProcess`, and `stage_reference_particles(reference: Path, destination: Path, case_ids: list[str], domains: list[str]) -> list[Path]`.
- CLI: `python scripts/shapeworks_completion.py --workshop ROOT --reference ROOT --allow-reference-fallback`.
- Consumes: Tasks 1 and 3 inventory/materialization/provenance interfaces.

- [ ] **Step 1: Write failing particle and fallback tests**

Use miniature fixtures to test local/world naming, exactly 256 finite XYZ rows, missing/malformed particles, identical case/domain order, controlled retry count, fallback staging isolation, and rejection of mismatched reference cohorts.

- [ ] **Step 2: Verify current failures**

Run: `conda run -n BIOF3001 pytest -q tests/test_shapeworks_completion.py`

Expected: FAIL because the completion module is absent.

- [ ] **Step 3: Implement the ShapeWorks boundary**

Implement vendor executable discovery, version capture, logged execution, checkpoint-aware retry, strict particle validation, and provenance-labeled fallback staging. Do not add an alignment/normalization stage.

- [ ] **Step 4: Attempt full Workshop regeneration**

Run the full CLI against `/Applications/ShapeWorks/bin/shapeworks`. Monitor it to completion and validate all expected local/world particles. If it meets a design-defined failure condition, diagnose, perform one controlled retry, then stage only validated matching reference particles if the retry fails.

- [ ] **Step 5: Verify published ShapeWorks artifacts**

Run the focused tests and the full particle validator. Expected: 71 × 15 local plus 71 × 15 world particle files, all finite with 256 XYZ rows, and an explicit source in the provenance manifest.

- [ ] **Step 6: Record the checkpoint**

Apply the Git/no-Git checkpoint rule; never stage generated particles or workbooks.

### Task 5: Standardized Full-Cohort PCA Exports

**Files:**
- Create: `scripts/prepare_full_pca_exports.py`
- Modify: `scripts/03_import_pca_exports.py`
- Create: `tests/test_pca_exports.py`
- Create at run time: `outputs/pca_exports/`
- Create at run time: `outputs/pca.npz`

**Interfaces:**
- Produces: `output_relative_path(value: str | Path) -> Path`, standardized mean/eigenvalue/unit-eigenvector/raw-score exports, and `pca.npz` keys `case_ids`, `mean`, `modes`, `eigenvalues`, and `scores`.
- Consumes: Task 4 validated particles and Task 3 case/domain order.

- [ ] **Step 1: Write failing PCA/export tests**

Test Windows/POSIX/mixed separators, shuffled case rejection, non-finite particle rejection, unit/orthogonal eigenvectors, score reconstruction tolerance, and deterministic domain concatenation.

- [ ] **Step 2: Verify current failures**

Run: `conda run -n BIOF3001 pytest -q tests/test_pca_exports.py`

Expected: FAIL because the preparation script is absent and mixed-path handling is incomplete.

- [ ] **Step 3: Implement standardized export preparation**

Port the reference-compatible export path normalization and full-cohort exporter, preserving raw component scores and verified case/domain order.

- [ ] **Step 4: Generate and verify PCA artifacts**

Run export preparation and import for the full cohort. Validate dimensions, orthonormality, finite arrays, reconstruction tolerance, and the expected standardized export inventory.

- [ ] **Step 5: Record the checkpoint**

Apply the Git/no-Git checkpoint rule; do not stage generated PCA artifacts.

### Task 6: PLS Training, Prediction, and Reconstruction

**Files:**
- Modify: `scripts/04_train_pls.py`
- Modify: `scripts/05_reconstruct.py`
- Create: `tests/test_regression_reconstruction.py`
- Create at run time: `outputs/regression/`
- Create at run time: `outputs/reconstructions/`

**Interfaces:**
- Consumes: Task 2 algorithms, Task 5 `pca.npz`, Task 3 case metadata/centroids.
- Produces: cross-validation results for `(1, 2, 3, 5, 8)`, selected/final model artifacts, predicted score tables, predicted concatenated particles, and 15 ordered reconstructed domains per held-out case.

- [ ] **Step 1: Write failing integration tests**

Test split isolation, supported component selection, deterministic case mapping, predicted-score dimensions, 3,840-particle reconstruction, 15-domain splitting, and non-finite rejection.

- [ ] **Step 2: Verify current failures**

Run: `conda run -n BIOF3001 pytest -q tests/test_regression_reconstruction.py`

Expected: FAIL on current PLS/reconstruction limitations or absent full artifacts.

- [ ] **Step 3: Complete the training/reconstruction scripts**

Use the course split and selection rule, persist cross-validation/selected-model metadata, and validate every mapping before writing predictions/reconstructions.

- [ ] **Step 4: Run and verify full regression/reconstruction**

Execute scripts 04 and 05 in `BIOF3001`; validate all held-out cases, all arrays, per-domain counts, and reports.

- [ ] **Step 5: Record the checkpoint**

Apply the Git/no-Git checkpoint rule; do not stage generated models or reconstructions.

### Task 7: Multi-PCA Model Suite and Interactive Viewer

**Files:**
- Modify: `scripts/07_interactive.py`
- Create: `scripts/build_pca_suite.py`
- Create: `tests/test_interactive_models.py`
- Create at run time: `outputs/pca_models/`
- Create at run time: `outputs/interactive_model.joblib`
- Create at run time: `outputs/interactive_particles.ply`

**Interfaces:**
- Produces: model discovery for K=3, K=4, K=5, and K=6 where supported, PCA-mode switching, run/suite directory selection, and headless export.
- Consumes: Tasks 5 and 6 PCA/regression/reconstruction outputs.

- [ ] **Step 1: Write failing viewer/model tests**

Test discovery of complete finite models, exclusion of partial/non-finite models, model switching, PCA-mode selection, missing optional GUI dependencies, and headless PLY export.

- [ ] **Step 2: Verify current failures**

Run: `conda run -n BIOF3001 pytest -q tests/test_interactive_models.py`

Expected: FAIL because suite building/mode switching is absent.

- [ ] **Step 3: Implement model-suite parity**

Move the reference suite builder into `scripts/`, enhance the viewer to discover and switch validated models/modes, and retain the existing default CLI behavior.

- [ ] **Step 4: Build and verify the model suite**

Generate available K=3/4/5/6 models, run all focused tests, export a headless PLY, and launch the GUI far enough to confirm model loading without claiming visual inspection if display automation is unavailable.

- [ ] **Step 5: Record the checkpoint**

Apply the Git/no-Git checkpoint rule; do not stage generated model files.

### Task 8: Resumable End-to-End Runner, Documentation, and Final Verification

**Files:**
- Create: `scripts/08_complete_pipeline.py`
- Modify: `README.md`
- Modify: `instructions/03_shapeworks.md`
- Modify: `instructions/05_interactive.md`
- Modify: `outputs/README.md`
- Create: `tests/test_complete_pipeline.py`
- Create at run time: `outputs/completion_report.md`

**Interfaces:**
- CLI: `python scripts/08_complete_pipeline.py --workshop ROOT --reference ROOT --allow-reference-fallback --resume`.
- Consumes: Tasks 1–7 public interfaces.
- Produces: stage status/checkpoints, one final provenance manifest, one completion report, and a nonzero exit status for any incomplete/invalid artifact family.

- [ ] **Step 1: Write failing runner tests**

Test stage ordering, resume after a completed stage, stop on validation failure, propagation of fallback permission, accurate source summaries, and nonzero exit for an incomplete model suite.

- [ ] **Step 2: Verify current failures**

Run: `conda run -n BIOF3001 pytest -q tests/test_complete_pipeline.py`

Expected: FAIL because the runner is absent.

- [ ] **Step 3: Implement the orchestration and update documentation**

Add the stage runner without duplicating stage logic. Document exact Conda/ShapeWorks commands, output families, resume behavior, validation, and fallback disclosure.

- [ ] **Step 4: Run the complete verification suite**

Run:

`conda run -n BIOF3001 pytest -q`

`conda run -n BIOF3001 python scripts/check_exercises.py`

`conda run -n BIOF3001 python scripts/08_complete_pipeline.py --workshop . --reference '/Users/wanglihang/Desktop/yr 3 sem1/BIOF3001/BIOF3001_Spine' --allow-reference-fallback --resume`

Expected: all tests/checks pass; the runner exits zero; reports identify the particle source and enumerate any fallback files.

- [ ] **Step 5: Compare completion against the reference**

Compare runnable capabilities and artifact families, not binary equality. Verify complete mesh/particle/PCA/regression/reconstruction/model/report families and document intentional differences.

- [ ] **Step 6: Perform the final Git safety check**

If Git is available, run `git status --short`, verify ignored/generated files are not staged, commit code/tests/docs, and push. If the target remains non-Git, do not create an empty commit or push; state this explicitly in the completion report and final response.

## Self-Review Result

- Spec coverage: recovery, raw-data preservation, code parity, ShapeWorks retry/fallback, PCA, regression, reconstruction, model suite, provenance, reporting, and end-to-end verification are each assigned to a task.
- Step scan: every task has a failing-test, implementation, verification, and checkpoint cycle.
- Type consistency: Tasks 3–8 consume the interfaces defined by Tasks 1–7 without renaming them.
- Review focus: all five high-risk conditions are pinned to explicit tests.
- Proportion: the plan defines interfaces and checks without transcribing implementation bodies.
