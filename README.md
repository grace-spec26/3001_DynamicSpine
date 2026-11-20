# BIOF3001: reconstruct a spine from 15 landmarks

Supervisor: Pengyu Lu

## Completed end-to-end pipeline

This Workshop now includes the full ShapeWorks-particle, whole-spine PCA, PLS,
reconstruction, K=3–6 model-suite, and interactive-viewer workflow. On this
machine, run it in the requested Conda environment:

```bash
conda run -n BIOF3001 python scripts/08_complete_pipeline.py \
  --workshop . \
  --reference '/Users/wanglihang/Desktop/yr 3 sem1/BIOF3001/BIOF3001_Spine' \
  --allow-reference-fallback \
  --resume
```

`--resume` validates each existing stage before skipping it. If a stage is
missing or invalid, the runner executes that stage and validates its outputs.
Reference fallback is never silent: `outputs/provenance_manifest.json` records
the source of every major artifact family, and `outputs/completion_report.md`
summarizes the final run.

The completed run used the Workshop's own centroids and workbooks. Its iCloud
raw-mesh materialization stalled before ShapeWorks could safely optimize the
entire cohort, so the 71 × 15 local/world particle set was validated and copied
from reference commit `830a10d`. PCA, PLS, reconstructions, and model suites were
then regenerated locally from those particles.

Start with supplied oriented vertebral meshes and segmentation masks. Extract 15 vertebral centroids, learn a statistical shape model in ShapeWorks, predict
its PCA scores with partial least squares regression, and reconstruct particles.
The final output is a point representation of T1–L3, not a triangle mesh.

## What is supplied

- 71 cases: masks only, 15 aligned surface meshes per case, and merged spines for viewing.
- One mask-to-aligned-mesh transform per case and an anatomical label map.
- A fixed 57/14 regression split and an eight-case GUI practice subset.
- A one-case Excel example, Python scaffolds, small checks, and exercise instructions.

Fifteen of the original 86 cases were excluded because at least one target label
touched a scan boundary. All included cases contain labels 8–22. This automated
screen reduces cropping risk; it is not a clinical certification of completeness.
The detailed screening results are in `data/cohort_audit.csv`.

## Setup

If you downloaded the code-only Moodle ZIP, first follow `DATA_DOWNLOAD.md` to
download the separate data archive from the HKU OneDrive link posted on Moodle.
The `data` folder must be alongside `scripts` before you run the checks below.

Use Python 3.11 or 3.12. From this folder, run:

```powershell
python -m venv .venv
.venv\Scripts\python.exe -m pip install -r requirements.txt
.venv\Scripts\python.exe scripts/00_check_package.py --hashes
```

Below, `python` means the interpreter in that environment. On Windows you can
always use `.venv\Scripts\python.exe` explicitly; activation is optional.
On macOS/Linux use `.venv/bin/python`. ShapeWorks Studio is installed separately;
use the same version as the instructor and record that version in your report.
No GPU, CT intensity volume, or Python ShapeWorks binding is needed for the scripts.
Allow several GB of RAM for full-size masks and additional disk space for GUI outputs.
The requirement ranges describe compatibility, not an exact locked environment.
Record your installed versions with `python -m pip freeze > outputs/environment.txt`.

## Exercise sequence

Edit **`scripts/student_tasks.py`**. Its seven TODO functions are the assessed work.
The numbered scripts provide file loading, argument parsing, saving and plotting.

| Step                        | Run                                                          | Read first                       |
| --------------------------- | ------------------------------------------------------------ | -------------------------------- |
| Inspect package             | `python scripts/00_check_package.py`                         | `instructions/01_data.md`        |
| Extract one case            | `python scripts/01_extract_centroids.py --case sub-verse502` | Exercise 1                       |
| Extract all cases           | `python scripts/01_extract_centroids.py`                     | Exercise 1                       |
| Build practice workbook     | `python scripts/02_create_project.py --subset demo`          | Exercise 2                       |
| Build full workbook         | `python scripts/02_create_project.py --subset all`           | Exercise 2                       |
| Optimize and export PCA     | ShapeWorks GUI                                               | `instructions/03_shapeworks.md`  |
| Import standardized exports | `python scripts/03_import_pca_exports.py --modes 3`          | Export contract                  |
| Fit and evaluate PLS        | `python scripts/04_train_pls.py`                             | Exercise 3                       |
| Reconstruct points          | `python scripts/05_reconstruct.py`                           | Exercise 4                       |
| Visualize                   | `python scripts/06_visualize.py`                             | Exercise 4                       |
| Explore live predictions    | `python scripts/07_interactive.py`                           | `instructions/05_interactive.md` |
| Build K=3–6 model suite     | `python scripts/build_pca_suite.py`                          | `instructions/05_interactive.md` |
| Validate/resume everything | `python scripts/08_complete_pipeline.py --workshop . --reference <path> --allow-reference-fallback --resume` | This README |

Run `python scripts/check_exercises.py` during development. Initially it reports
NOT IMPLEMENTED and exits with code 1. This is expected until you complete the TODOs.

The optional live viewer provides landmark sliders, a rotatable 3D prediction,
PC scores, vertebral visibility controls, reset and PLY export. Install its
desktop dependencies with `python -m pip install -r requirements_interactive.txt`.
It uses **your trained model and your completed functions**. Run training again
to create `outputs/interactive_model.joblib` if you trained with an earlier package.
On this machine, launch the reference-style native panel with:

```bash
conda run -n BIOF3001 python scripts/07_interactive.py
```

When the validated experiment suite exists, the panel starts with K=3 and its
selector offers K=3, 4, 5, and 6. The panel exposes 15 three-dimensional
landmarks (45 X/Y/Z coordinate controls); the fitted PLS model predicts the
displayed PC scores and spine. These controls are not direct PC-coefficient
sliders.

Each selectable experiment must contain a matching `interactive_model.joblib`
and `pca.npz` in its canonical `outputs/pca_models/pca_03` through `pca_06`
folder. Missing, corrupt, or incompatible experiments are rejected rather than
mixed into the active PCA frame. Reset restores the starting landmarks, while
Save PLY always exports all 3,840 predicted points—even levels hidden in the
viewer.

For prediction without a desktop window, use the same validated model path in
headless mode:

```bash
conda run -n BIOF3001 python scripts/07_interactive.py \
  --headless \
  --run-dir outputs/pca_models/pca_03 \
  --output outputs/interactive_particles.ply
```

The eight-case subset is for practicing the GUI, not for the final regression.
Use the full-cohort exports for the PLS stage. Optimization time depends on hardware;
the workshop can demonstrate that stage while the group completes it later.

## Folder guide

`data/masks` contains unchanged `.nii.gz` masks. `data/meshes` contains separate
T1–L3 meshes. `data/merged_spines` is for display only. `data/transforms` contains
the provided coordinate maps. `templates` contains the workbook example.
`outputs` is where your scripts and GUI project should write results.

The data split measures **regression performance within a shared shape model**.
All cases may contribute to ShapeWorks correspondence/PCA in this introductory
exercise. Do not describe the test numbers as external validation of the entire
pipeline. See the extension in the exercise guide for a stricter research design.

Read `DATA_SOURCES.md` for attribution and data terms. The scientific starting
point is [VerSe](https://github.com/anjany/verse); the supplied preprocessing is
Pengyu's previous work. Do not include raw data in your code repository.
