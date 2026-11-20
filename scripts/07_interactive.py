"""Provided real-time viewer. Complete student_tasks.py first; no GUI TODOs.

Uses your saved PLS estimator and PCA basis, not the research demo weights.
Input: aligned millimetres, T1–L3. See instructions/05_interactive.md.
"""
from pathlib import Path
import argparse
import sys
from dataclasses import dataclass
import numpy as np
from common import OUTPUT, LEVELS
from model_validation import load_validated_model_bundle
from student_tasks import landmark_features, predict_scores, reconstruct_particles

SEGMENTS = tuple(LEVELS)
COLORS = ("#0B1026", "#0E1E41", "#123A6F", "#165792", "#1B6FAE",
          "#228CC3", "#37B7D6", "#62D5DA", "#8FE6C3", "#C2F0B3",
          "#F4F1A1", "#FEE07A", "#FDBA4A", "#F36B3E", "#C81D4F")


@dataclass(frozen=True)
class ViewerPaths:
    model_path: Path
    suite_dir: Path
    output_path: Path
    fixed_output: bool
    fallback_model_path: Path | None = None


def resolve_viewer_paths(model_path, run_dir, suite_dir, output_path, headless):
    """Resolve viewer artifacts while preserving explicit CLI precedence."""
    run_dir_was_explicit = run_dir is not None
    run_dir = Path(run_dir) if run_dir is not None else OUTPUT
    suite_dir = Path(suite_dir) if suite_dir is not None else OUTPUT/'pca_models'
    fallback_model_path = None
    if model_path is not None:
        model_path = Path(model_path)
    elif not headless and not run_dir_was_explicit:
        preferred = suite_dir/'pca_03'/'interactive_model.joblib'
        fallback = run_dir/'interactive_model.joblib'
        if preferred.is_file():
            model_path = preferred
            fallback_model_path = fallback
        else:
            model_path = fallback
    else:
        model_path = run_dir/'interactive_model.joblib'
    fixed_output = output_path is not None
    if fixed_output:
        output_path = Path(output_path)
    elif headless:
        output_path = run_dir/'interactive_particles.ply'
    else:
        output_path = model_path.parent/'interactive_particles.ply'
    return ViewerPaths(
        model_path, suite_dir, output_path, fixed_output, fallback_model_path
    )

class StudentSpineModel:
    def __init__(self, path, pca_path=None):
        # Load only model files produced by your own training script.
        try:
            data = load_validated_model_bundle(path, pca_path)
        except Exception as error:
            raise ValueError(f'Inconsistent model bundle; rerun training: {error}') from error
        self.model = data['model']
        self.mean = np.asarray(data['mean'])
        self.modes = np.asarray(data['modes'])
        self.eigenvalues = np.asarray(data.get('eigenvalues', np.ones(len(self.modes))))
        self.case_ids = list(map(str, data['case_ids']))
        self.centres = np.asarray(data['centres'])
        self.default_case = data['default_case']

    def predict(self, keypoints):
        keypoints = np.asarray(keypoints, dtype=float)
        if keypoints.shape != (15, 3) or not np.isfinite(keypoints).all():
            raise ValueError('Expected finite (15,3) aligned landmarks in millimetres.')
        # Identical preprocessing to training; do not normalize the curve here.
        features = landmark_features(keypoints[None, ...])
        scores = np.asarray(predict_scores(self.model, features))
        if scores.shape != (1, len(self.modes)) or not np.isfinite(scores).all():
            raise ValueError('predict_scores must return finite (1,K) scores.')
        particles = np.asarray(reconstruct_particles(scores, self.mean, self.modes))
        if particles.shape != (1,3840,3) or not np.isfinite(particles).all():
            raise ValueError('reconstruct_particles must return finite (1,3840,3).')
        return scores[0], particles[0]

    def split(self, particles):
        return np.split(particles, 15)

def save_particles(path, particles):
    """Write an ASCII PLY that opens in standard point-cloud viewers."""
    path = Path(path).expanduser().resolve()
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="ascii", newline="\n") as output:
        output.write("ply\nformat ascii 1.0\n")
        output.write(f"element vertex {len(particles)}\n")
        output.write("property double x\nproperty double y\nproperty double z\nend_header\n")
        np.savetxt(output, particles, fmt="%.12g")
    return path


def _canonical_mode_count(model_path, suite_dir):
    """Return the K encoded by a canonical pca_03 ... pca_06 path."""
    model_path = Path(model_path).resolve()
    suite_dir = Path(suite_dir).resolve()
    for mode_count in (3, 4, 5, 6):
        canonical = suite_dir/f'pca_{mode_count:02d}'/'interactive_model.joblib'
        if model_path == canonical.resolve():
            return mode_count
    return None


def discover_models(model_path, suite_dir, case=None):
    """Load only complete, finite model bundles from a PCA experiment suite."""
    active_path = Path(model_path).resolve()
    active_pca = active_path.parent/'pca.npz'
    active = StudentSpineModel(active_path, active_pca)
    expected_active_modes = _canonical_mode_count(active_path, suite_dir)
    if (expected_active_modes is not None
            and len(active.modes) != expected_active_modes):
        raise ValueError(
            f'Canonical pca_{expected_active_modes:02d} model expected '
            f'{expected_active_modes} PCA modes, got {len(active.modes)}.'
        )
    selected_case = case or active.default_case
    if selected_case not in active.case_ids:
        raise ValueError(f'Unknown case {selected_case}; use a case from the PCA cohort.')
    active_centres = active.centres[active.case_ids.index(selected_case)]
    choices = {}
    for expected_modes in (3, 4, 5, 6):
        folder = Path(suite_dir)/f'pca_{expected_modes:02d}'
        path = folder/'interactive_model.joblib'
        if not path.is_file() or not (folder/'pca.npz').is_file():
            continue
        try:
            candidate = StudentSpineModel(path, folder/'pca.npz')
            if len(candidate.modes) != expected_modes:
                continue
            if selected_case not in candidate.case_ids:
                continue
            candidate_centres = candidate.centres[
                candidate.case_ids.index(selected_case)
            ]
            if not np.allclose(candidate_centres, active_centres, atol=1e-6):
                continue
            if not np.allclose(candidate.mean, active.mean, atol=1e-6):
                continue
        except Exception:
            continue
        choices[expected_modes] = path.resolve()
    choices[len(active.modes)] = active_path
    return active, dict(sorted(choices.items()))


def load_gui_models(paths, case=None):
    """Load the automatic K=3 choice, falling back to the root run atomically."""
    try:
        model, choices = discover_models(paths.model_path, paths.suite_dir, case)
        return model, choices, Path(paths.model_path).resolve()
    except (OSError, ValueError, KeyError):
        if paths.fallback_model_path is None:
            raise
        fallback = Path(paths.fallback_model_path).resolve()
        model, choices = discover_models(fallback, paths.suite_dir, case)
        return model, choices, fallback


class InteractiveSession:
    """Own the atomic, GUI-independent state of one interactive prediction."""

    def __init__(self, model, model_path, case, output_path, fixed_output,
                 model_loader=None):
        if case not in model.case_ids:
            raise ValueError(f'Unknown case {case}; use a case from the PCA cohort.')
        self.model = model
        self.model_path = Path(model_path).resolve()
        self.case = case
        self.output_path = Path(output_path)
        self.fixed_output = bool(fixed_output)
        self.model_loader = model_loader or StudentSpineModel
        self.initial_keypoints = model.centres[model.case_ids.index(case)].copy()
        self.keypoints = self.initial_keypoints.copy()

    def predict(self):
        return self.model.predict(self.keypoints)

    def edit_point(self, point, coordinate, value):
        value = float(value)
        if not np.isfinite(value):
            raise ValueError('Landmark coordinates must be finite.')
        self.keypoints[point, coordinate] = value

    def reset(self):
        self.keypoints[...] = self.initial_keypoints

    def switch_model(self, path, expected_mode_count=None):
        path = Path(path).resolve()
        pca_path = path.parent/'pca.npz'
        candidate = self.model_loader(path, pca_path)
        if (expected_mode_count is not None
                and len(candidate.modes) != expected_mode_count):
            raise ValueError(
                f'Selected experiment expected {expected_mode_count} PCA modes, '
                f'got {len(candidate.modes)}.'
            )
        if self.case not in candidate.case_ids or not np.allclose(
                candidate.centres[candidate.case_ids.index(self.case)],
                self.initial_keypoints, atol=1e-6):
            raise ValueError('This model uses different input landmarks for this case.')
        if not np.allclose(candidate.mean, self.model.mean, atol=1e-6):
            raise ValueError('This model uses a different mean shape or PCA frame.')
        prediction = candidate.predict(self.keypoints)
        self.model = candidate
        self.model_path = path
        if not self.fixed_output:
            self.output_path = path.parent/'interactive_particles.ply'
        return prediction

    def save(self, path=None):
        target = Path(path) if path is not None else self.output_path
        _, particles = self.predict()
        return save_particles(target, particles)


def attempt_session_switch(session, path, expected_mode_count=None):
    """Switch atomically or return the retained model's current prediction."""
    try:
        prediction = session.switch_model(path, expected_mode_count)
    except (OSError, ValueError, KeyError) as error:
        return False, session.predict(), str(error)
    return True, prediction, None


def launch_gui(model, keypoints, output_path, model_path=None, model_choices=None,
               case=None, fixed_output=False):
    # GUI dependencies are deliberately optional for CLI/headless use.
    from PyQt5 import QtCore, QtWidgets
    import pyvista as pv
    from pyvistaqt import QtInteractor

    session = InteractiveSession(
        model, model_path, case, output_path, fixed_output
    )

    class CurveAndPCQtApp(QtWidgets.QMainWindow):
        def __init__(self):
            super().__init__()
            self.setWindowTitle("BIOF3001 — your landmark-to-spine model")
            self.resize(1400, 850)
            self.session = session
            self.views_ready = False
            self.controls = []
            self.actors = []
            central = QtWidgets.QWidget()
            self.setCentralWidget(central)
            layout = QtWidgets.QHBoxLayout(central)
            panel = QtWidgets.QWidget()
            panel.setMaximumWidth(480)
            panel_layout = QtWidgets.QVBoxLayout(panel)
            for text in (
                "15 aligned landmarks (T1–L3): edit X / Y / Z in mm",
                "Your PLS + PCA model • edits explore predictions, not test accuracy",
            ):
                label = QtWidgets.QLabel(text)
                label.setWordWrap(True)
                panel_layout.addWidget(label)
            model_row = QtWidgets.QHBoxLayout()
            model_row.addWidget(QtWidgets.QLabel("PCA modes"))
            self.pca_selector = QtWidgets.QComboBox()
            for mode_count, path in (
                model_choices or {len(model.modes): self.session.model_path}
            ).items():
                self.pca_selector.addItem(str(mode_count), str(path))
            active_index = self.pca_selector.findData(str(self.session.model_path))
            if active_index >= 0:
                self.pca_selector.setCurrentIndex(active_index)
            self.pca_selector.setEnabled(False)
            self.pca_selector.currentIndexChanged.connect(self.change_model)
            model_row.addWidget(self.pca_selector)
            panel_layout.addLayout(model_row)
            scroll = QtWidgets.QScrollArea()
            scroll.setWidgetResizable(True)
            fields = QtWidgets.QWidget()
            grid = QtWidgets.QGridLayout(fields)
            for j, text in enumerate(("Point", "X", "Y", "Z")):
                grid.addWidget(QtWidgets.QLabel(text), 0, j)
            # Slider ticks represent 0.01 mm; coordinates remain in the training frame.
            lower = self.session.keypoints.min(axis=0) - 30.0
            upper = self.session.keypoints.max(axis=0) + 30.0
            self.timer = QtCore.QTimer(self)
            self.timer.setSingleShot(True)
            self.timer.setInterval(40)
            self.timer.timeout.connect(self.update_views)
            for i, name in enumerate(SEGMENTS):
                grid.addWidget(QtWidgets.QLabel(name), i + 1, 0)
                for j in range(3):
                    cell = QtWidgets.QWidget()
                    cell_layout = QtWidgets.QVBoxLayout(cell)
                    cell_layout.setContentsMargins(0, 0, 0, 0)
                    spin = QtWidgets.QDoubleSpinBox()
                    spin.setDecimals(2)
                    spin.setRange(float(lower[j]), float(upper[j]))
                    spin.setSingleStep(0.1)
                    spin.setValue(float(self.session.keypoints[i, j]))
                    slider = QtWidgets.QSlider(QtCore.Qt.Horizontal)
                    slider.setRange(int(np.floor(lower[j] * 100)), int(np.ceil(upper[j] * 100)))
                    slider.setValue(int(round(self.session.keypoints[i, j] * 100)))
                    slider.valueChanged.connect(
                        lambda value, p=i, c=j: self.edit_point(p, c, value / 100))
                    spin.valueChanged.connect(
                        lambda value, p=i, c=j: self.edit_point(p, c, value))
                    cell_layout.addWidget(spin)
                    cell_layout.addWidget(slider)
                    grid.addWidget(cell, i + 1, j + 1)
                    self.controls.append((spin, slider))
            scroll.setWidget(fields)
            panel_layout.addWidget(scroll)
            buttons = QtWidgets.QHBoxLayout()
            for title, callback in (("Reset", self.reset_keypoints), ("Save PLY…", self.save_current)):
                button = QtWidgets.QPushButton(title)
                button.clicked.connect(callback)
                buttons.addWidget(button)
            panel_layout.addLayout(buttons)
            visibility = QtWidgets.QGroupBox("Visible vertebrae")
            checks = QtWidgets.QGridLayout(visibility)
            self.checkboxes = []
            for i, name in enumerate(SEGMENTS):
                checkbox = QtWidgets.QCheckBox(name)
                checkbox.setChecked(True)
                checkbox.toggled.connect(self.update_visibility)
                checks.addWidget(checkbox, i // 5, i % 5)
                self.checkboxes.append(checkbox)
            panel_layout.addWidget(visibility)
            self.status = QtWidgets.QLabel()
            self.status.setWordWrap(True)
            panel_layout.addWidget(self.status)
            layout.addWidget(panel, 2)
            self.left = QtInteractor(central, auto_update=False, multi_samples=0)
            self.right = QtInteractor(central, auto_update=False, multi_samples=0)
            for plotter in (self.left, self.right):
                plotter.set_background("white")
                plotter.add_axes()
                layout.addWidget(plotter, 2)
            self.status.setText("Preparing predicted particles...")

        def initialize_views(self):
            print("Initializing 3D views...", flush=True)
            self.update_views(initial=True)
            self.views_ready = True
            self.pca_selector.setEnabled(True)
            print(
                "3D views ready: input landmarks on the left, prediction on the right.",
                flush=True,
            )

        def change_model(self, index):
            if not self.views_ready:
                return
            path = Path(self.pca_selector.itemData(index))
            if path.resolve() == self.session.model_path:
                return
            self.timer.stop()
            expected_mode_count = int(self.pca_selector.itemText(index))
            switched, prediction, error = attempt_session_switch(
                self.session, path, expected_mode_count
            )
            if not switched:
                self.pca_selector.blockSignals(True)
                self.pca_selector.setCurrentIndex(
                    self.pca_selector.findData(str(self.session.model_path)))
                self.pca_selector.blockSignals(False)
                self.update_views(prediction=prediction)
                self.status.setText(f'Model switch failed: {error}')
                return
            self.update_views(prediction=prediction)

        def edit_point(self, point, coordinate, value):
            self.session.edit_point(point, coordinate, value)
            spin, slider = self.controls[point * 3 + coordinate]
            for widget in (spin, slider):
                widget.blockSignals(True)
            spin.setValue(float(value))
            slider.setValue(int(round(value * 100)))
            for widget in (spin, slider):
                widget.blockSignals(False)
            self.timer.start()

        def reset_keypoints(self):
            self.timer.stop()
            self.session.reset()
            for i in range(15):
                for j in range(3):
                    spin, slider = self.controls[i * 3 + j]
                    for widget in (spin, slider):
                        widget.blockSignals(True)
                    value = float(self.session.keypoints[i, j])
                    spin.setValue(value)
                    slider.setValue(int(round(value * 100)))
                    for widget in (spin, slider):
                        widget.blockSignals(False)
            self.update_views()

        def update_visibility(self):
            for actor, checkbox in zip(self.actors, self.checkboxes):
                actor.SetVisibility(checkbox.isChecked())
            self.right.render()

        def update_views(self, initial=False, prediction=None):
            if prediction is None:
                try:
                    prediction = self.session.predict()
                except ValueError as error:
                    self.status.setText(str(error))
                    return
            scores, particles = prediction
            if initial:
                self.keypoint_cloud = pv.PolyData(self.session.keypoints.copy())
                self.curve = pv.lines_from_points(self.session.keypoints)
                self.left.add_points(
                    self.keypoint_cloud,
                    color="crimson",
                    point_size=10,
                    render_points_as_spheres=False,
                    render=False,
                )
                self.left.add_mesh(
                    self.curve, color="gray", line_width=2, render=False
                )
                self.left.add_text(
                    "Input landmarks (aligned mm)",
                    font_size=10,
                    color="black",
                    render=False,
                )
                self.right.add_text(
                    "Predicted particles (PCA frame, mm)",
                    font_size=10,
                    color="black",
                    render=False,
                )
                self.clouds = []
                for points, color in zip(self.session.model.split(particles), COLORS):
                    cloud = pv.PolyData(points.copy())
                    self.clouds.append(cloud)
                    self.actors.append(
                        self.right.add_points(
                            cloud,
                            color=color,
                            point_size=5,
                            render_points_as_spheres=False,
                            render=False,
                        )
                    )
                for plotter in (self.left, self.right):
                    plotter.view_yz()
                    plotter.reset_camera()
            else:
                self.keypoint_cloud.points = self.session.keypoints.copy()
                self.curve.points = self.session.keypoints.copy()
                for cloud, points in zip(
                    self.clouds, self.session.model.split(particles)
                ):
                    cloud.points = points.copy()
            self.left.render()
            self.right.render()
            self.status.setText(f"PCA modes: {len(scores)} • {len(particles):,} particles • PCs: " +
                                ", ".join(f"{value:.3f}" for value in scores))

        def save_current(self):
            filename, _ = QtWidgets.QFileDialog.getSaveFileName(
                self,
                "Save all 15 reconstructed vertebrae",
                str(self.session.output_path),
                "Point cloud (*.ply)",
            )
            if filename:
                try:
                    path = self.session.save(filename)
                    self.status.setText(f"Saved all 15 vertebrae: {path}")
                except (OSError, ValueError) as error:
                    self.status.setText(f"Save failed: {error}")

        def closeEvent(self, event):
            self.timer.stop()
            self.left.close()
            self.right.close()
            event.accept()

    app = QtWidgets.QApplication.instance() or QtWidgets.QApplication([sys.argv[0]])
    window = CurveAndPCQtApp()
    window.show()
    QtCore.QTimer.singleShot(0, window.initialize_views)
    return app.exec_()


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--model',type=Path,default=None)
    parser.add_argument('--case',help='Start from this case; defaults to the first held-out case')
    parser.add_argument('--headless',action='store_true',help='Export without GUI dependencies')
    parser.add_argument('--output',type=Path,default=None)
    parser.add_argument('--run-dir',type=Path,default=None,
                        help='Directory containing one PCA experiment')
    parser.add_argument('--suite-dir',type=Path,default=None,
                        help='Parent directory containing pca_03, pca_04, etc.')
    args=parser.parse_args(argv)
    paths = resolve_viewer_paths(
        args.model, args.run_dir, args.suite_dir, args.output, args.headless
    )
    args.model = paths.model_path
    args.output = paths.output_path
    try:
        if args.headless:
            pca_path = args.model.parent/'pca.npz'
            model=StudentSpineModel(args.model, pca_path)
            choices={len(model.modes):args.model}
        else:
            model,choices,active_model_path=load_gui_models(paths,args.case)
            args.model=active_model_path
            if not paths.fixed_output:
                args.output=active_model_path.parent/'interactive_particles.ply'
        case=args.case or model.default_case
        if case not in model.case_ids:
            raise ValueError(f'Unknown case {case}; use a case from the PCA cohort.')
        points=model.centres[model.case_ids.index(case)].copy()
        scores,particles=model.predict(points)
        print(f'Starting case: {case}; aligned input in mm; {len(scores)} PCA scores')
        if args.headless:
            print(save_particles(args.output,particles))
            print('PC scores:',scores)
            return 0
        return launch_gui(model,points,args.output,args.model,choices,case,
                          fixed_output=paths.fixed_output)
    except ImportError as error:
        parser.exit(1,f'{error}\nInstall requirements_interactive.txt for the GUI, or use --headless.\n')
    except NotImplementedError as error:
        parser.exit(1,f'{error}\nComplete Exercises 3 and 4 in student_tasks.py first.\n')
    except (OSError,ValueError,KeyError) as error:
        parser.exit(1,f'{error}\nRun 04_train_pls.py after completing your functions and PCA import.\n')

if __name__=='__main__':
    sys.exit(main())
