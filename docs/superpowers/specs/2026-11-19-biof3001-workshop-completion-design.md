# BIOF3001 Workshop Completion Design

## Intent

Complete `BIOF3001_Spine_Workshop` to the functional and artifact completeness of the finished `BIOF3001_Spine` reference while retaining the Workshop's own raw cohort as the primary scientific source. The preferred result is a fully reproducible ShapeWorks-to-PCA-to-regression/model pipeline generated from the Workshop data. Reference artifacts are a last-resort fallback, not the default source.

The completed project is for a BIOF3001 workshop user who needs to run, inspect, and demonstrate the entire dynamic-spine analysis rather than only the fundamental algorithms.

## Current State

- The Workshop contains the full 71-case cohort: 71 masks, 71 merged spines, 71 transforms, and 1,065 vertebral meshes across 15 domains.
- Its numbered preprocessing, PCA, regression, reconstruction, and interactive-viewer scripts largely match the reference structure.
- `student_tasks.py` contains the Workshop's own algorithm implementations, but its PLS validation prevents the supplied exercise checks and planned cross-validation from accepting valid component counts.
- Existing Workshop outputs are partial: centroids are present, groomed meshes are missing 14 files, only one case has local/world particles, and the PCA, regression, reconstruction, and final model suites are absent.
- Many existing data/output files are iCloud dataless placeholders and must be materialized before they can be processed reliably.
- ShapeWorks is installed under `/Applications/ShapeWorks`, while the requested Conda environment is `BIOF3001`; the ShapeWorks Python package is not currently importable directly from that environment.
- The Workshop directory is not currently a Git repository. Therefore the requested commit/push workflow cannot be performed there unless a repository and remote are supplied; generated scientific outputs would be excluded from Git in any case.

## Goals

1. Preserve the Workshop's raw data and its student-authored algorithms wherever they are scientifically correct.
2. Repair code and configuration blockers needed to execute the full course pipeline.
3. Regenerate the missing ShapeWorks particles from the Workshop's meshes when the installed ShapeWorks tooling can do so reliably.
4. Generate standardized full-cohort PCA exports, cross-validated PLS regression, predicted PCA scores/particles, reconstructions, model assets, reports, and interactive-viewer inputs.
5. Match the reference project's expected capabilities and artifact categories, while allowing regenerated numeric results to differ from the reference.
6. Record the source and validation status of every major artifact so regenerated and fallback results cannot be confused.
7. Run the full pipeline in the `BIOF3001` Conda environment except for ShapeWorks commands that must use the vendor installation/runtime.

## Non-goals

- Do not change or fabricate the Workshop's raw masks, transforms, or meshes.
- Do not force binary or numeric identity with the reference outputs.
- Do not redesign the underlying scientific method or introduce a different correspondence, PCA, or regression framework.
- Do not commit model weights, generated datasets, uploads, or generated outputs.
- Do not silently replace a failed Workshop artifact with a reference artifact.

## Anti-goals

The work is considered unsuccessful if it appears complete only because reference outputs were copied without disclosure, if raw Workshop data are overwritten, if partial iCloud files are treated as valid artifacts, or if the pipeline cannot identify which stages were regenerated and which used fallback data.

## Constraints and Working Assumptions

- Target project: `/Users/wanglihang/Desktop/yr 3 sem1/BIOF3001/3D dynamic spine/BIOF3001_Spine_Workshop`.
- Completed reference: `/Users/wanglihang/Desktop/yr 3 sem1/BIOF3001/BIOF3001_Spine`.
- Primary shell/runtime: `conda run -n BIOF3001 ...` or an activated `BIOF3001` shell.
- ShapeWorks may be invoked through `/Applications/ShapeWorks/bin/shapeworks`, `ShapeWorksStudio`, or its bundled Python runtime when the Conda environment cannot import the package.
- The reference cohort is assumed to correspond to the same case/domain naming scheme. This must be verified by case IDs, domain names, particle dimensions, and ordering before any fallback copy.
- Long-running ShapeWorks optimization is permitted, but it must emit logs and checkpoints so failure is diagnosable and reruns are resumable.
- Since the target is not a Git repository, a timestamped, same-volume backup of source/configuration and the existing partial output tree will provide recovery. No remote push will be claimed.

## Approaches Considered

### 1. Regenerate first in place, with provenance-aware fallback — selected

Complete the existing Workshop directory, preserving its raw data and algorithms. Regenerate each stage in dependency order. Only after a concrete stage fails despite diagnostics and a controlled retry may matching reference artifacts be used. Record fallback use in a manifest and rerun all downstream stages locally whenever possible.

Tradeoff: this provides the strongest scientific provenance and the desired complete project, but ShapeWorks optimization can be slow and vendor-runtime integration may need care.

### 2. Build a separate completed sibling copy

Perform the same regeneration in a new directory and leave the Workshop untouched.

Tradeoff: safer isolation, but it leaves the named Workshop project incomplete and creates two competing versions.

### 3. Copy the completed reference outputs immediately

Graft the reference output tree into the Workshop and make only path/configuration changes.

Tradeoff: fastest, but it does not demonstrate that the Workshop's raw data generated the results and makes provenance weak. It is retained only as the fallback mechanism inside approach 1.

## Design

### 1. Recovery and inventory

Before modifying source or outputs:

- Capture a timestamped inventory containing paths, sizes, iCloud/dataless flags, and hashes where files are materialized.
- Preserve the current source, configuration, instructions, manifests, and partial output tree in a timestamped backup on the same volume. Prefer atomic moves or APFS cloning to avoid needless downloads and duplication.
- Leave `data/` untouched.
- Add or update `.gitignore` so virtual environments, caches, `.DS_Store`, model weights, and generated data/output directories are excluded if the directory later becomes a Git repository.

### 2. Code parity without erasing Workshop work

Bring the Workshop's orchestration and usability features up to reference parity:

- Fix `student_tasks.py` validation so PLS accepts the feature dimensions used by the supplied checks and accepts component counts supported by scikit-learn's PLS implementation and the course's cross-validation grid. Retain the Workshop's actual algorithm bodies unless verification proves them incorrect.
- Add robust validation to reconstruction and related interfaces where the reference expects it.
- Port the reference project's unique-mesh-name project-generation option and checksum validation.
- Add the full PCA-export preparation script, multi-model build support, and the enhanced interactive model/PCA-mode selection behavior.
- Bring exercise checks, README/instructions, and command examples into alignment with the completed pipeline and platform-neutral paths.
- Avoid copying reference instructions as executable authority; they are treated as project documentation and evidence, subordinate to this design and the user's request.

### 3. Data materialization and preprocessing

- Verify `cases.csv` contains all 71 cases and all required columns.
- Materialize only the raw files required by the next stage, failing early on zero-byte, placeholder, or unreadable inputs.
- Regenerate or verify all 1,065 prealigned/groomed meshes. Existing valid files may be reused after checksum/readability checks; the 14 missing files must be regenerated.
- Recompute or verify all 71 centroid JSON files against the corresponding input geometry.
- Produce a uniquely named ShapeWorks input set so the 15 domains for every case remain unambiguous.

### 4. ShapeWorks regeneration and fallback boundary

Attempt a full-cohort ShapeWorks optimization using the course configuration:

- 71 subjects and 15 fixed domains per subject.
- 256 particles per domain, giving 3,840 particles per subject when concatenated.
- Use the prealigned meshes without adding an unintended Procrustes, scale, or pose normalization stage.
- Save the generated project/workbook, optimizer logs, runtime/version information, checkpoints, and local/world particle files.

A regeneration attempt is considered failed only when one of these occurs and remains after one controlled retry with corrected runtime/path/materialization issues:

- the installed CLI/runtime cannot load the generated project;
- optimization terminates with a reproducible error;
- required inputs cannot be read after materialization;
- optimization completes but expected particle files are missing, malformed, non-finite, or dimensionally inconsistent;
- the result fails case/domain ordering or particle-count validation.

If full ShapeWorks regeneration fails:

1. Validate that the reference particles cover the identical 71 case IDs and 15 ordered domains with 256 particles per domain.
2. Copy only the required matching ShapeWorks particle/input artifacts into a distinct fallback staging area.
3. Record the source reference path and, where available, checksum and reference commit in `outputs/provenance_manifest.json`.
4. Use those particles as the input to freshly run Workshop PCA, regression, reconstruction, and model-building stages. Copy downstream reference outputs only if a downstream local stage independently fails its own validation.

This preserves the maximum amount of locally regenerated work even when the correspondence stage cannot be reproduced.

### 5. PCA exports

Generate the whole-spine shape-and-pose PCA from the complete particle matrix:

- preserve case and domain order from `cases.csv` and the project workbook;
- export the mean particle vector, eigenvalues, unit eigenvectors, raw component scores, predicted-score tables, and a compact `pca.npz` bundle;
- create the standardized `outputs/pca_exports` package expected by later scripts;
- export the modes required by the completed viewer/model suite, including the reference-compatible K=3, K=4, K=5, and K=6 configurations where supported by the retained variance/components;
- verify centering, dimensional consistency, orthonormal eigenvectors, score reconstruction, finite values, and deterministic ordering.

### 6. Regression and reconstruction

- Use the course regression split in `cases.csv`.
- Cross-validate PLS component counts using the course grid, select the supported optimum using the reference selection rule, fit the final model, and generate held-out predictions.
- Save model parameters/summary, predicted PCA scores, per-case predicted particles, and quantitative reports.
- Reconstruct predicted spine models from the mean, eigenvectors, and predicted scores; split concatenated particles back into the 15 domains in the verified order.
- Verify array dimensions, absence of NaN/Inf values, deterministic case mapping, and reconstruction consistency.

### 7. Model and viewer outputs

- Build the interactive model assets and PLY/mesh outputs needed by `07_interactive.py`.
- Build the multi-PCA model suite and allow the viewer to switch among available PCA configurations/modes.
- Generate reference-equivalent summary/comparison images and reports where the necessary rendering dependencies are available.
- Smoke-test model discovery and loading programmatically. GUI rendering will also receive a launch test; any display-only limitation will be documented rather than treated as a successful visual inspection.

### 8. Provenance and completion report

Create:

- `outputs/provenance_manifest.json`: source classification (`workshop_regenerated`, `workshop_reused`, or `reference_fallback`) for every major artifact family, commands, environment versions, timestamps, checksums/counts, and failure reasons;
- `outputs/completion_report.md`: commands run, validation results, artifact counts, selected PLS configuration, PCA/model variants, fallback use, and remaining caveats;
- persistent stage logs under `outputs/logs/`.

The final report must explicitly say whether ShapeWorks was regenerated successfully and enumerate any reference-derived files.

## Expected Artifact Families

Completion is judged by capability and validated inventory rather than byte-for-byte equality. The finished Workshop should contain, at minimum:

- complete groomed/ShapeWorks input meshes for 71 × 15 domains;
- complete local and world particle sets for 71 × 15 domains;
- 71 centroid records;
- full project/workbook files and ShapeWorks logs;
- standardized PCA exports, PCA scores/eigenvectors/eigenvalues, mean particles, and `pca.npz`;
- PLS model selection/results and predicted scores;
- predicted particle files and reconstructed per-domain outputs for the held-out cohort;
- interactive model/PLY assets and the requested multi-PCA model suite;
- reports, comparison figures where renderable, provenance manifest, and completion report.

## Verification Strategy

Verification proceeds after each stage and again end to end:

1. Run the supplied exercise checker in `BIOF3001`; all implemented tasks must pass.
2. Run targeted tests for PLS component-grid compatibility, reconstruction shapes, unique mesh naming, path normalization, and fallback provenance.
3. Import/compile all scripts and exercise their `--help` or dry-run paths where available.
4. Validate exact case/domain inventories and expected artifact dimensions/counts.
5. Parse representative and boundary files rather than trusting their presence; reject dataless, empty, malformed, or non-finite outputs.
6. Verify PCA orthonormality and numerical reconstruction tolerances.
7. Verify regression outputs map to the intended held-out cases.
8. Smoke-test all model variants and viewer discovery/loading.
9. Run the complete pipeline or its resumable stage runner from a clean output location to prove reproducibility.
10. Compare artifact families and runnable capabilities with the completed reference and document intentional differences.

No completion claim will be made solely from file existence or a zero exit status from an earlier stage.

## Builder Decisions Left Open

The implementer may choose the exact timestamp/backup naming, log format details, retry command flags, and whether to use APFS clone or atomic rename based on filesystem support. The implementer may also add small test helpers or a stage runner where doing so improves resumability, provided the scientific method and provenance rules above do not change.

