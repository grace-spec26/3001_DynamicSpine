# Reference Interactive Panel Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Make a no-argument Workshop launch reproduce the reference landmark-to-spine GUI with a validated K=3/4/5/6 selector and reliable native rendering.

**Architecture:** Keep `scripts/07_interactive.py` as the public CLI/GUI entry point, but extract pure launch-path and session-state decisions inside that module so they can be tested without Qt/VTK. Port the reference window lifecycle and rendering settings while retaining `model_validation.py` for every model/PCA load.

**Tech Stack:** Python 3.10, NumPy, scikit-learn/joblib, PyQt5, PyVista, pyvistaqt, pytest; Conda environment `BIOF3001`.

**Spec:** `docs/superpowers/specs/2026-11-20-reference-interactive-panel-design.md`

## Global Constraints

- Use existing Workshop artifacts in `outputs/pca_models/pca_03` through `pca_06`; never copy reference weights or outputs.
- Preserve strict paired validation of every `interactive_model.joblib` and companion `pca.npz`.
- Preserve headless operation without importing PyQt5, PyVista, or pyvistaqt.
- Do not add direct PC coefficient sliders.
- Native GUI uses deferred first render, `auto_update=False`, `multi_samples=0`, plain points, and explicit renders.
- The Workshop is not a Git repository: run the required Git detection, preserve a timestamped source backup, and report that commit/push are unavailable rather than modifying the separate reference repository.

## Review Focus

- K=3 is missing or invalid: GUI falls back to the requested/default run model and does not crash during path resolution; covered in Task 1.
- A suite candidate is individually valid but uses a different case/landmark/mean frame: omit it from the selector and reject it if selected; covered in Tasks 1–2.
- User supplies `--model`, `--run-dir`, `--suite-dir`, or `--output`: explicit values retain precedence, and explicit output remains fixed across switches; covered in Tasks 1–2.
- A switch fails after landmark edits: current model, current edits, selected-combo state, displayed prediction, and output path remain unchanged; covered in Task 2.
- A vertebra is hidden when saving: the PLY still contains all 3,840 finite predicted coordinates; covered in Task 2.

---

### Task 1: Deterministic launch paths and compatible suite discovery

**Files:**
- Modify: `scripts/07_interactive.py`
- Modify: `tests/test_interactive_models.py`

**Interfaces:**
- Consumes: `StudentSpineModel(path, pca_path=None)` and `load_validated_model_bundle(model_path, pca_path)`.
- Produces: `ViewerPaths` dataclass; `resolve_viewer_paths(model_path, run_dir, suite_dir, output_path, headless) -> ViewerPaths`; `discover_models(model_path, suite_dir, case=None) -> tuple[StudentSpineModel, dict[int, Path]]`.

- [ ] **Step 1: Write failing path-resolution tests**

Add tests asserting that GUI defaults select `outputs/pca_models/pca_03`, headless defaults retain `outputs/interactive_model.joblib`, missing K=3 falls back to the run directory, explicit model/suite/output paths win, and `fixed_output` is recorded.

- [ ] **Step 2: Run the new path tests and verify RED**

Run: `conda run -n BIOF3001 pytest -q tests/test_interactive_models.py -k 'resolve_viewer_paths'`

Expected: FAIL because `resolve_viewer_paths`/`ViewerPaths` do not exist.

- [ ] **Step 3: Implement launch resolution**

Implement `ViewerPaths` and `resolve_viewer_paths(...)` with the exact precedence from the design. Change `--suite-dir` to default to `None`; call the resolver before loading models.

- [ ] **Step 4: Run path tests and verify GREEN**

Run: `conda run -n BIOF3001 pytest -q tests/test_interactive_models.py -k 'resolve_viewer_paths'`

Expected: all selected tests PASS.

- [ ] **Step 5: Write failing compatible-discovery tests**

Use shared case IDs, centres, and mean arrays across positive K=3/4/5/6 fixtures. Assert exact choices `[3, 4, 5, 6]`; then create an individually valid candidate with a mismatched mean or starting landmarks and assert it is omitted for the active case.

- [ ] **Step 6: Run discovery tests and verify RED**

Run: `conda run -n BIOF3001 pytest -q tests/test_interactive_models.py -k 'discovery'`

Expected: at least the compatibility test FAILS because current discovery does not compare the active case/frame.

- [ ] **Step 7: Implement compatible discovery**

Update `discover_models(model_path, suite_dir, case=None)` to validate the active bundle first, choose `case or active.default_case`, strictly validate each paired candidate, and include it only when case membership, case landmarks, and mean frame match.

- [ ] **Step 8: Run the interactive-model test file**

Run: `conda run -n BIOF3001 pytest -q tests/test_interactive_models.py`

Expected: all tests PASS.

- [ ] **Step 9: Record checkpoint**

Run `git rev-parse --is-inside-work-tree`; when it reports no repository, record the passing command and files in this plan's SDD ledger instead of committing.

### Task 2: Reference interaction state and native window lifecycle

**Files:**
- Modify: `scripts/07_interactive.py`
- Modify: `tests/test_interactive_models.py`

**Interfaces:**
- Consumes: `ViewerPaths`, compatible choices from Task 1, `StudentSpineModel.predict`, and `save_particles`.
- Produces: `InteractiveSession(model, model_path, case, output_path, fixed_output, model_loader=StudentSpineModel)` with `edit_point`, `reset`, `switch_model`, `predict`, and `save` methods; `launch_gui` delegates state transitions to it.

- [ ] **Step 1: Write failing session tests**

Add tests that edit one landmark, switch K=3→K=6 while retaining the edit, reset to the original landmarks, and preserve model/path/output/prediction after an incompatible switch raises. Test both fixed and model-relative output-path behavior.

- [ ] **Step 2: Run session tests and verify RED**

Run: `conda run -n BIOF3001 pytest -q tests/test_interactive_models.py -k 'interactive_session'`

Expected: FAIL because `InteractiveSession` does not exist.

- [ ] **Step 3: Implement `InteractiveSession`**

Implement state transitions atomically: validate and predict with a candidate before assigning any session field. `save()` always predicts and writes all 3,840 particles, independent of GUI visibility.

- [ ] **Step 4: Run session tests and verify GREEN**

Run: `conda run -n BIOF3001 pytest -q tests/test_interactive_models.py -k 'interactive_session'`

Expected: all selected tests PASS.

- [ ] **Step 5: Strengthen the PLY test first**

Change the headless test to parse all PLY coordinates, assert shape `(3840, 3)`, finiteness, and equality to `StudentSpineModel.predict` for the same case. Run it before GUI edits to establish the numerical boundary.

- [ ] **Step 6: Port the reference native lifecycle and UI behavior**

In `launch_gui`, add the reference's maximum-width/word-wrapped panel, `views_ready`, disabled selector until initialization, deferred `initialize_views`, explicit-render QtInteractors, plain point actors, status markers, and clean close. Delegate edit/reset/switch/save state to `InteractiveSession`; on switch failure restore the combo and leave current clouds unchanged.

- [ ] **Step 7: Run all interactive tests**

Run: `conda run -n BIOF3001 pytest -q tests/test_interactive_models.py`

Expected: all tests PASS and headless mode never imports `PyQt5`.

- [ ] **Step 8: Record checkpoint**

Record the test evidence in the SDD ledger; no Git commit is possible in the Workshop.

### Task 3: User-facing launch contract and end-to-end verification

**Files:**
- Modify: `README.md`
- Modify: `instructions/05_interactive.md`
- Modify: `tests/test_complete_pipeline.py` only if the validated default path changes the completion contract

**Interfaces:**
- Consumes: no-argument launch behavior and session/UI contract from Tasks 1–2.
- Produces: exact launch instructions and final verification evidence.

- [ ] **Step 1: Update launch documentation**

Document that `conda run -n BIOF3001 python scripts/07_interactive.py` starts K=3 and exposes K=3/4/5/6; clarify that the controls edit landmarks and the model predicts PC scores rather than exposing direct PC coefficient sliders.

- [ ] **Step 2: Run all automated verification**

Run:

```bash
conda run -n BIOF3001 pytest -q
conda run -n BIOF3001 python scripts/check_exercises.py
conda run -n BIOF3001 python -m py_compile scripts/*.py
conda run -n BIOF3001 python scripts/07_interactive.py --headless --run-dir outputs/pca_models/pca_03 --output outputs/reference_panel_headless_check.ply
```

Expected: complete pytest suite passes, course checks report 6/6 PASS, compilation exits 0, and headless output reports three PC scores and 3,840 finite vertices.

- [ ] **Step 3: Run native GUI interaction smoke**

Launch `conda run -n BIOF3001 python scripts/07_interactive.py`, require `Initializing 3D views...` then a line beginning `3D views ready`, and verify the window remains responsive. Exercise one landmark edit, K=3→K=6, Reset, hide one level, Save PLY, and normal close. Validate the saved PLY still has 3,840 finite coordinates. If direct automation cannot drive native widgets, document this as a manual native smoke and retain the automated session-state/PLY tests.

- [ ] **Step 4: Run final Git detection**

Run `git status --short`. If it fails with `not a git repository`, do not stage generated outputs and do not commit/push the separate reference repository; report that exact limitation.

- [ ] **Step 5: Final whole-change review and ledger**

Create the final review package, obtain one fresh review under `superpowers:requesting-code-review`, fix Critical/Important findings once with RED→GREEN tests, defer Minor findings in the ledger, and finish under `superpowers:finishing-a-development-branch` adapted to the no-Git target.
