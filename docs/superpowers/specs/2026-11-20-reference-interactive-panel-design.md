# Reference Interactive Panel Design

## Intent

Make `python scripts/07_interactive.py` in the Workshop open the same useful
interactive experience as the completed `BIOF3001_Spine` reference: edit the
15 aligned vertebral landmarks, select K=3/4/5/6 PCA experiments, and see the
predicted whole-spine particles update live. The Workshop's stricter artifact
validation remains in force so matching the reference UI does not weaken the
completed pipeline.

## Goals

- A no-argument native launch starts the validated K=3 Workshop model when the
  K=3–6 suite is present.
- The left control panel contains the reference's 15 T1–L3 rows with X/Y/Z
  numeric fields and sliders, a K=3/4/5/6 selector, Reset, Save PLY, vertebral
  visibility controls, and live PC/status text.
- The middle view shows the edited landmarks and connecting curve; the right
  view shows 3,840 predicted particles, coloured by vertebral level.
- Model switching preserves the active case, input landmarks, and shared PCA
  frame, and rejects incomplete, corrupt, or incompatible bundles.
- The native macOS window initializes rendering only after it is visible, using
  the reference's explicit-render settings to avoid the prior startup problem.
- Headless prediction remains supported and produces the same numeric result as
  the selected model's GUI path.

## Non-goals

- Do not add direct PC1/PC2 coefficient sliders; approach 1 reproduces the
  reference interaction, where landmark edits drive predicted PCA scores.
- Do not copy reference model weights or overwrite Workshop PCA/PLS artifacts.
- Do not change PCA training, PLS training, reconstruction mathematics, or raw
  Workshop data.
- Do not make the GUI a clinical or external-validation tool.

## Constraints

- Run with the existing `BIOF3001` Conda environment and existing PyQt5,
  PyVista, and pyvistaqt dependencies.
- Use `outputs/pca_models/pca_03` through `pca_06`; each selection must pass the
  shared strict joblib/PCA validator before it is offered or loaded.
- Preserve headless operation without importing GUI dependencies.
- The Workshop directory is not a Git repository; changes are made in place,
  backed up, tested, and reported without inventing a commit or push target.

## Approaches considered

1. **Reference behavior with Workshop safeguards (selected).** Port the
   reference lifecycle, layout, and default model discovery, while retaining
   strict bundle validation and the Workshop's `outputs/pca_models` layout.
   This matches the visible behavior without regressing artifact integrity.
2. **Literal reference file copy.** Closest textual match, but it discards the
   Workshop's stricter validation and assumes a different experiment layout.
3. **Extended PCA coefficient editor.** Adds direct mode-amplitude sliders, but
   is intentionally excluded because it is not the requested exact reference
   behavior.

## Design

### Launch and model discovery

Extract a pure path-resolution function. For a no-argument GUI launch, prefer
`outputs/pca_models/pca_03/interactive_model.joblib` and use
`outputs/pca_models` as the suite. Explicit `--model`, `--run-dir`, and
`--suite-dir` continue to override defaults. Headless mode keeps its explicit
model/run-directory semantics and never imports Qt/VTK.

Discovery loads every candidate through `load_validated_model_bundle` together
with its companion `pca.npz`. Only complete K=3,4,5,6 models whose starting
landmarks and mean frame match the active launch model appear. The active model
is validated by the same path. If the preferred K=3 file is absent, retain the
reference-compatible fallback to the requested/default run directory instead of
silently selecting an unrelated file.

Explicit `--output` remains fixed when switching models. Without it, each model
uses its own experiment folder. The existing headless default is preserved so a
headless call does not unexpectedly change from the caller's selected run.

### Window lifecycle and controls

Match the reference layout and startup sequence:

- fixed maximum-width control panel with word-wrapped guidance;
- selector disabled until both native render windows are visible;
- `QtInteractor(auto_update=False, multi_samples=0)` for both views;
- `QTimer.singleShot(0, initialize_views)` after `window.show()`;
- plain OpenGL points rather than sphere impostors;
- explicit `render()` after edits and visibility changes.

The 45 landmark controls remain synchronized without recursive signals. Changes
are debounced by 40 ms, passed through the trained feature/PLS/PCA path, and
update both the curve and predicted particle clouds.

### Model switching

The selector stops pending updates, strictly loads the selected bundle with its
companion PCA file, checks the case, original landmarks, and mean frame, runs a
prediction smoke test, then updates the view. Failure restores the prior
selection and displays a status message. Successful switching retains current
landmark edits and changes the default save path to the selected experiment.

### Saving and shutdown

Save PLY always exports all 3,840 predicted particles, including hidden levels.
Reset restores the launch-case landmarks. Closing stops the debounce timer and
closes both VTK interactors cleanly.

## Verification

- Test path resolution: no-argument GUI launch selects K=3 and discovers exactly
  K=3/4/5/6 from the Workshop suite; explicit paths retain precedence.
- Test model discovery and switching boundaries with valid, corrupt, non-finite,
  and frame-mismatched bundles. Compatible test bundles share one landmark/mean
  fixture across K values.
- Test headless output by reading all 3,840 coordinates and requiring finiteness,
  not only by checking the PLY header.
- Run the complete pytest suite and course checks in `BIOF3001`.
- Launch the real native GUI and require the log markers `Initializing 3D
  views...` and the prefix `3D views ready`; verify one landmark edit, switch
  K=3→K=6 while retaining that edit, Reset, hide one level, and save. The saved
  file must still equal all 3,840 coordinates predicted by the selected model.
  Close the window normally to exercise timer/interactor cleanup. If automated
  native interaction is unavailable, capture these as a documented manual native
  smoke while keeping the headless coordinate test automated.

## Acceptance criteria

Running this from the Workshop folder:

```bash
conda run -n BIOF3001 python scripts/07_interactive.py
```

opens the reference-style panel with editable T1–L3 landmark controls, an
enabled selector containing 3, 4, 5, and 6, two working 3D views, live predicted
particle updates, Reset/visibility/Save controls, and no startup crash.
