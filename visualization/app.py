"""Configure, preview, run and inspect Barnes–Hut simulations with Qt."""

from concurrent.futures import ThreadPoolExecutor
from datetime import datetime
import copy
import html
import json
import os
from pathlib import Path
import re
import shutil
import struct
import sys
import tempfile
import time

import numpy as np
from PySide6 import QtCore, QtGui, QtWidgets
import pyqtgraph as pg
import pyqtgraph.opengl as gl

from viewer import BACKGROUND, create_viewer, directory_frames, frame_bounds, load_frames, read_frame

ROOT = Path(__file__).resolve().parents[1]
MODELS = {"galaxy": "Stellar disk + bulge", "uniform3d": "Uniform 3D cube",
          "gaussian2d": "Gaussian 2D disk", "gaussian3d": "Gaussian 3D cloud"}
VELOCITIES = {"equilibrium": "Galaxy Jeans estimates", "rest": "At rest",
              "random": "Random · virialized", "rotation": "Rotation · approximate"}
DESCRIPTIONS = {
    "galaxy": "Exponential disk, sech² thickness and a Plummer bulge. Approximate Jeans velocities; no dark-matter halo.",
    "uniform3d": "Uniform positions in a cube. Scale is its half-width. This is a cloud model, not an equilibrium galaxy.",
    "gaussian2d": "Independent Gaussian X/Y coordinates, with Z = 0 locally. Scale is coordinate standard deviation σ.",
    "gaussian3d": "Independent Gaussian X/Y/Z coordinates. Scale is coordinate standard deviation σ.",
}
OBJECT_TIPS = {
    "particles": "Number of numerical particles in this object. They share its total mass equally. More particles resolve finer structure but increase simulation and output costs.",
    "mass": "Total mass of this object, not mass per particle. 1 M0 = 100 billion solar masses. Mass affects the gravity and initialized internal velocities.",
    "height": "Vertical scale z0 in the disk profile sech²(z/z0). 1 L0 = 10 kpc. This is a scale height, not a sharp upper/lower boundary.",
    "bulge_radius": "Plummer scale radius of the spherical bulge, in L0. It controls central concentration and is not the bulge's outer edge.",
    "bulge_fraction": "Fraction of this object's mass assigned to the bulge. The remaining mass is in the disk. Equal particle masses also make this approximately the bulge particle fraction.",
    "q": "Sets the disk's radial random-motion target at 2.2 disk scale lengths, before global virial scaling. Larger Q gives a warmer disk. It does not guarantee exact equilibrium.",
    "seed": "Random seed for this object. Matching settings and simulator reproduce the same sample. Use a different seed for an independently sampled companion galaxy.",
}
SCALE_TIPS = {
    "galaxy": ("Disk scale length (L0)", "Exponential radial scale Rd: disk surface density falls as exp(-R/Rd). 1 L0 = 10 kpc. This is not an outer radius."),
    "uniform3d": ("Cube half-width (L0)", "Positions are sampled uniformly from -a to +a along each local axis, then centered. This value is a; the cube's side length is 2a. 1 L0 = 10 kpc."),
    "gaussian2d": ("Standard deviation (L0)", "Standard deviation σ of each independent local X/Y Gaussian coordinate. Local Z is zero. This is a spatial Gaussian, not a Gaussian radius. 1 L0 = 10 kpc."),
    "gaussian3d": ("Standard deviation (L0)", "Standard deviation σ of each independent local X/Y/Z Gaussian coordinate. The cloud has no sharp outer boundary. 1 L0 = 10 kpc."),
}
MOTION_TIPS = {
    "equilibrium": "Approximate Jeans estimates for a rotating disk and pressure-supported bulge. Includes random motion and internal virial normalization; not exact equilibrium.",
    "rest": "Zero internal velocity. Only the bulk velocity below is added. A self-gravitating cloud at rest will generally collapse.",
    "random": "Random internal velocities, scaled using this object's softened forces to global virial balance. This does not make an arbitrary density profile an exact equilibrium.",
    "rotation": "Approximate rotation about the object's local Z axis, normalized to internal virial balance. A toy rotating cloud, not an equilibrium galaxy model.",
}


class CompactDoubleSpinBox(QtWidgets.QDoubleSpinBox):
    def textFromValue(self, value):
        text = super().textFromValue(value)
        point = self.locale().decimalPoint()
        return text.rstrip("0").rstrip(point) if point in text else text


def default_object(number=1):
    return {"name": f"Galaxy {number}", "model": "galaxy", "velocity_mode": "equilibrium",
            "particles": 1000, "mass": 1.0, "scale": .3, "height": .06,
            "bulge_radius": .1, "bulge_fraction": .2, "q": 1.5, "seed": 41 + number,
            "position": [0.0, 0.0, 0.0], "velocity": [0.0, 0.0, 0.0], "rotation": [0.0, 0.0, 0.0]}


def write_scene(path, objects):
    """Small, versioned text bridge to the dependency-free C++ executable."""
    lines = ["NBODY_SCENE 1"]
    for obj in objects:
        values = [obj["model"], obj["velocity_mode"], obj["particles"], obj["mass"],
                  obj["scale"], obj["height"], obj["bulge_radius"], obj["bulge_fraction"],
                  obj["q"], obj["seed"], *obj["position"], *obj["velocity"], *obj["rotation"]]
        name = obj["name"]
        if any(ord(c) < 32 for c in name):
            raise ValueError("Object names must not contain control characters")
        quoted = '"' + name.replace('\\', '\\\\').replace('"', '\\"') + '"'
        lines.append(" ".join(str(value) for value in values) + " " + quoted)
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def simulation_arguments(configuration):
    values = configuration["simulation"]
    arguments = [argument for key, flag in (("threads", "--threads"), ("steps", "--steps"),
             ("dt", "--dt"), ("theta", "--theta"), ("softening", "--softening"),
             ("snapshot_interval", "--snapshot-interval")) for argument in (flag, str(values[key]))]
    if values.get("frames") is not None:
        arguments += ["--frames", str(values["frames"])]
    arguments += ["--integrator", values.get("integrator", "euler")]
    return arguments


def snapshot_directory(path):
    return path / "snapshots" if (path / "snapshots").is_dir() else path


def read_json(path):
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"Expected a JSON object in {path.name}")
    return value


def load_display(path, preview=False):
    directory = snapshot_directory(path)
    metadata_path = directory / "metadata.json"
    metadata = read_json(metadata_path) if metadata_path.exists() else {}
    frames = load_frames(directory, metadata)
    if preview:
        # One preview frame only: detach it from the temporary file so replacing
        # a preview is also safe on platforms that lock memory-mapped files.
        frames = np.array(read_frame(frames, 0), copy=True)[None, ...]
    bounds = frame_bounds(frames)
    if "particle_count" in metadata and metadata["particle_count"] != bounds[0]:
        raise ValueError("Snapshot particle count disagrees with metadata.json")
    return frames, metadata, bounds


def velocity_vectors(path, limit=256):
    with path.open("rb") as file:
        if file.read(8) != b"NBODYIC1":
            raise ValueError("Unrecognized prepared state")
        count, = struct.unpack("<Q", file.read(8))
        if count == 0 or path.stat().st_size != 16 + count * 56:
            raise ValueError("Invalid prepared state size")
        data = np.fromfile(file, dtype="<f8").reshape(count, 7)
    if not np.isfinite(data).all():
        raise ValueError("Prepared state contains nonfinite values")
    indices = np.linspace(0, count - 1, min(count, limit), dtype=int)
    positions, velocities = data[indices, :3], data[indices, 3:6]
    return np.stack((positions, positions + velocities * .05), axis=1).reshape(-1, 3)


class SimulationApp(QtWidgets.QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Barnes–Hut · simulation studio")
        self.resize(1550, 980)
        self.window_escape = QtGui.QShortcut(QtGui.QKeySequence("Escape"), self)
        self.window_escape.activated.connect(lambda: self.showNormal() if self.isFullScreen() else None)
        self.objects = [default_object()]
        self.editing = False
        self.process = None
        self.operation = None
        self.cancelled = False
        self.closing = False
        self.preview_temp = None
        self.preview_signature = None
        self.preparing_signature = None
        self.preview_view = None
        self.run_view = None
        self.run_view_path = None
        self.vector_item = None
        self.current_run = None
        self.log_file = None
        self.stdout_buffer = ""
        self.loader = ThreadPoolExecutor(max_workers=1, thread_name_prefix="snapshot-loader")
        self.loading_future = None
        self.setStyleSheet("QWidget { background: #0b111a; color: #dce7f3; }"
            "QLineEdit, QSpinBox, QDoubleSpinBox, QComboBox, QListWidget, QTableWidget, QPlainTextEdit {"
            " background: #111c29; border: 1px solid #2b3b50; padding: 4px; }"
            "QPushButton { background: #1a2c3e; border: 1px solid #34465a; padding: 6px 10px; }"
            "QPushButton:disabled { color: #526579; }"
            "QPushButton#primaryAction { background: #24778a; border-color: #65d9ed; font-weight: 600; }"
            "QPushButton#primaryAction:disabled { background: #152131; border-color: #2b3b50; color: #526579; }"
            "QGroupBox { margin-top: 12px; padding: 10px 6px 6px; border: 1px solid #24364c; border-radius: 4px; }"
            "QGroupBox::title { subcontrol-origin: margin; left: 10px; color: #9bb1c7; }"
            "QToolTip { background: #233548; color: #edf4fb; border: 1px solid #526579; padding: 6px; }"
            "QTabBar::tab { padding: 9px 18px; } QTabBar::tab:selected { color: #65d9ed; }"
            "QHeaderView::section { background: #152131; padding: 6px; }")
        workspace = QtWidgets.QWidget()
        layout = QtWidgets.QHBoxLayout(workspace)
        sidebar = QtWidgets.QWidget()
        sidebar.setMinimumWidth(360)
        sidebar.setMaximumWidth(440)
        side = QtWidgets.QVBoxLayout(sidebar)
        side.setContentsMargins(0, 0, 0, 0)
        self.sections = QtWidgets.QStackedWidget()
        self.navigation = QtWidgets.QButtonGroup(self)
        for index, name in enumerate(("Setup && Preview", "Simulation", "Saved Runs")):
            button = QtWidgets.QPushButton(name)
            button.setCheckable(True)
            button.setStyleSheet("QPushButton:checked { background: #24778a; border-color: #65d9ed; }")
            self.navigation.addButton(button, index)
            side.addWidget(button)
        self.navigation.idClicked.connect(self.sections.setCurrentIndex)
        self.sections.currentChanged.connect(lambda index: self.navigation.button(index).setChecked(True))
        side.addWidget(self.sections, 1)
        layout.addWidget(sidebar)
        self.display = QtWidgets.QStackedWidget()
        self.preview_placeholder = QtWidgets.QLabel("INITIAL CONDITIONS\n\nConfigure one or more objects, then generate a preview.")
        self.preview_placeholder.setAlignment(QtCore.Qt.AlignmentFlag.AlignCenter)
        self.display.addWidget(self.preview_placeholder)
        layout.addWidget(self.display, 1)
        self.setCentralWidget(workspace)
        self.build_setup()
        self.build_simulation()
        self.build_runs()
        self.object_list.setCurrentRow(0)
        self.poll_timer = QtCore.QTimer(self)
        self.poll_timer.timeout.connect(self.poll)
        self.poll_timer.start(50)
        self.refresh_runs()
        self.invalidate_preview()
        self.preview_notice.setText("Generate a preview before starting a simulation.")

    @staticmethod
    def button(layout, label, callback):
        button = QtWidgets.QPushButton(label)
        button.clicked.connect(callback)
        layout.addWidget(button)
        return button

    @staticmethod
    def number(low, high, value, decimals=None, step=None):
        widget = QtWidgets.QSpinBox() if decimals is None else CompactDoubleSpinBox()
        if decimals is not None:
            widget.setDecimals(decimals)
        widget.setRange(low, high)
        widget.setValue(value)
        widget.setKeyboardTracking(False)
        if step is not None:
            widget.setSingleStep(step)
        elif decimals is not None:
            widget.setStepType(QtWidgets.QAbstractSpinBox.StepType.AdaptiveDecimalStepType)
        return widget

    @staticmethod
    def field_row(form, text, widget, tooltip):
        label = QtWidgets.QLabel(text)
        tooltip = f"<qt>{html.escape(tooltip)}</qt>"
        label.setToolTip(tooltip)
        label.setBuddy(widget)
        widget.setToolTip(tooltip)
        widget.setAccessibleName(text)
        form.addRow(label, widget)
        return label

    @staticmethod
    def form_group(title, layout):
        group = QtWidgets.QGroupBox(title)
        form = QtWidgets.QFormLayout(group)
        form.setRowWrapPolicy(QtWidgets.QFormLayout.RowWrapPolicy.WrapLongRows)
        form.setVerticalSpacing(9)
        layout.addWidget(group)
        return group, form

    def build_setup(self):
        page = QtWidgets.QWidget()
        side = QtWidgets.QVBoxLayout(page)
        side.addWidget(QtWidgets.QLabel("OBJECTS"))
        self.object_list = QtWidgets.QListWidget()
        self.object_list.setMaximumHeight(110)
        self.object_list.setToolTip("Select the object to edit in Structure and Motion. Run settings apply to the entire scene.")
        self.object_list.addItem(self.objects[0]["name"])
        self.object_list.currentRowChanged.connect(self.load_editor)
        side.addWidget(self.object_list)
        row = QtWidgets.QHBoxLayout()
        self.add_object_button = self.button(row, "Add", self.add_object)
        self.duplicate_button = self.button(row, "Duplicate", self.duplicate_object)
        self.remove_button = self.button(row, "Remove", self.remove_object)
        self.add_object_button.setToolTip("Add an independently sampled object. Set its position and bulk velocity in Motion.")
        self.duplicate_button.setToolTip("Copy this object's settings with a new seed and a shifted position. Edit the copy before generating a preview.")
        self.remove_button.setToolTip("Remove the selected object from this setup. Saved runs are unaffected.")
        side.addLayout(row)

        editor_tabs = QtWidgets.QTabWidget()
        self.editor_tabs = editor_tabs
        object_page = QtWidgets.QWidget()
        object_layout = QtWidgets.QVBoxLayout(object_page)
        _, form = self.form_group("Distribution", object_layout)
        self.name = QtWidgets.QLineEdit()
        self.model = QtWidgets.QComboBox()
        for key, label in MODELS.items():
            self.model.addItem(label, key)
            self.model.setItemData(self.model.count() - 1, DESCRIPTIONS[key], QtCore.Qt.ItemDataRole.ToolTipRole)
        self.motion = QtWidgets.QComboBox()
        for key, label in VELOCITIES.items():
            self.motion.addItem(label, key)
            self.motion.setItemData(self.motion.count() - 1, MOTION_TIPS[key], QtCore.Qt.ItemDataRole.ToolTipRole)
        self.field_row(form, "Object name", self.name, "Label for this object in the setup and saved run metadata.")
        self.field_row(form, "Model", self.model, "Choose the local spatial distribution. Use Motion to place, orient and move the object in the scene.")
        self.description = QtWidgets.QLabel()
        self.description.setWordWrap(True)
        self.description.setStyleSheet("color: #8fa5bb;")
        form.addRow(self.description)
        self.galaxy_group, galaxy_form = self.form_group("Stellar disk and bulge", object_layout)
        self.fields = {}
        self.field_labels = {}
        for key, label, low, high, value, decimals in (
            ("particles", "Particles", 1, 2_000_000_000, 1000, None),
            ("mass", "Mass (M0)", .000001, 1e6, 1, 6),
            ("scale", "Scale / σ (L0)", .000001, 1e6, .3, 6),
            ("seed", "Random seed", 0, 2_147_483_647, 42, None),
            ("height", "Disk scale height (L0)", .000001, 1e6, .06, 6),
            ("bulge_radius", "Bulge radius (L0)", .000001, 1e6, .1, 6),
            ("bulge_fraction", "Bulge mass fraction", 0, 1, .2, 3),
            ("q", "Toomre Q target", .1, 5, 1.5, 2),
        ):
            widget = self.number(low, high, value, decimals)
            self.fields[key] = widget
            target = galaxy_form if key in ("height", "bulge_radius", "bulge_fraction", "q") else form
            tip = SCALE_TIPS["galaxy"][1] if key == "scale" else OBJECT_TIPS[key]
            self.field_labels[key] = self.field_row(target, label, widget, tip)
            widget.valueChanged.connect(self.edit_object)
        units = QtWidgets.QLabel("1 L0 = 10 kpc · 1 M0 = 10¹¹ solar masses")
        units.setWordWrap(True)
        units.setStyleSheet("color: #8fa5bb;")
        object_layout.addWidget(units)
        object_layout.addStretch()

        motion_page = QtWidgets.QWidget()
        motion_layout = QtWidgets.QVBoxLayout(motion_page)
        _, motion_form = self.form_group("Internal motion", motion_layout)
        self.field_row(motion_form, "Velocity model", self.motion, "Internal particle velocities are initialized locally before the bulk velocity and orientation are applied. Galaxy velocities are determined by the galaxy model.")
        self.motion_description = QtWidgets.QLabel()
        self.motion_description.setWordWrap(True)
        self.motion_description.setStyleSheet("color: #8fa5bb;")
        motion_form.addRow(self.motion_description)
        self.transform_fields = {}
        transform_tips = {
            "position": "Center of this object in world coordinates, in L0. 1 L0 = 10 kpc. Changing the center moves the whole object.",
            "velocity": "Velocity added to every particle in this object, in L0/T0. It moves the object as a whole and is not rescaled by internal virial normalization.",
            "rotation": "Right-handed rotations in degrees about the local X, Y and Z axes, applied in that order. Both positions and internal velocities are rotated before placement.",
        }
        for key, label in (("position", "Center (L0)"), ("velocity", "Bulk velocity (L0/T0)"),
                           ("rotation", "Rotation XYZ (degrees)")):
            group, transform_form = self.form_group(label, motion_layout)
            group.setToolTip(f"<qt>{html.escape(transform_tips[key])}</qt>")
            container = QtWidgets.QWidget()
            row = QtWidgets.QHBoxLayout(container)
            row.setContentsMargins(0, 0, 0, 0)
            widgets = []
            for axis in "XYZ":
                spin = self.number(-1e6 if key != "rotation" else -360,
                                   1e6 if key != "rotation" else 360, 0, 4, .1 if key != "rotation" else 5)
                spin.setPrefix(axis + " ")
                spin.setMinimumWidth(70)
                spin.setToolTip(f"<qt>{axis} component. {html.escape(transform_tips[key])}</qt>")
                spin.setAccessibleName(f"{label} {axis}")
                spin.valueChanged.connect(self.edit_object)
                widgets.append(spin)
                row.addWidget(spin)
            self.transform_fields[key] = widgets
            transform_form.addRow(container)
        motion_layout.addStretch()

        simulation_page = QtWidgets.QWidget()
        simulation_layout = QtWidgets.QVBoxLayout(simulation_page)
        _, time_form = self.form_group("Time integration", simulation_layout)
        _, output_form = self.form_group("Saved snapshots", simulation_layout)
        _, accuracy_form = self.form_group("Accuracy and parallelism", simulation_layout)
        self.output_mode = QtWidgets.QComboBox()
        self.output_mode.addItem("Number of frames", "frames")
        self.output_mode.addItem("Every N integration steps", "interval")
        self.field_row(output_form, "Save by", self.output_mode, "Choose an exact number of saved frames or a fixed step interval. Output sampling does not change the integration timestep or simulated duration.")
        self.frame_count = self.number(2, 2001, 201)
        self.frame_count_label = self.field_row(output_form, "Frames", self.frame_count,
            "Exact number of saved snapshots, including the initial and final states. Output times are spread evenly and rounded to integration steps. At most one frame per step; zero steps saves one frame. More frames increase disk usage and write time, not integration accuracy.")
        self.simulation_fields = {}
        self.simulation_labels = {}
        simulation_tips = {
            "threads": "OpenMP threads used by the simulator. More threads are not always faster. This also controls initial velocity calibration.",
            "steps": "Number of integration updates. Simulated duration is steps × timestep. Zero steps produces only the initial snapshot.",
            "dt": "Time advanced per integration step, in T0. Smaller values can improve accuracy but require more steps for the same duration. Output frame count does not affect this value.",
            "theta": "Barnes–Hut opening angle. Smaller values generally give more accurate, slower forces; zero opens the full tree. Also affects initial virial calibration.",
            "softening": "Plummer gravitational softening length, in L0. Reduces singular close-particle forces and changes small-scale dynamics. Also used when initializing velocities.",
            "snapshot_interval": "Save every N integration steps, plus the initial and final states. The final interval may be shorter. This preserves the sampling used by older saved setups.",
        }
        for key, label, low, high, value, decimals in (
            ("threads", "Threads", 1, 4096, min(14, os.cpu_count() or 1), None),
            ("steps", "Integration steps", 0, 2_000_000_000, 2000, None),
            ("dt", "Timestep (T0)", 1e-8, 10, .001, 8),
            ("theta", "Opening angle", 0, 2, .5, 3),
            ("softening", "Softening (L0)", 1e-8, 1e3, .02, 8),
            ("snapshot_interval", "Snapshot every N steps", 1, 2_000_000_000, 10, None),
        ):
            widget = self.number(low, high, value, decimals)
            self.simulation_fields[key] = widget
            target = time_form if key in ("steps", "dt") else (output_form if key == "snapshot_interval" else accuracy_form)
            self.simulation_labels[key] = self.field_row(target, label, widget, simulation_tips[key])
            widget.valueChanged.connect(self.output_settings_changed if key in ("steps", "dt", "snapshot_interval") else self.invalidate_preview)
        self.output_summary = QtWidgets.QLabel()
        self.output_summary.setWordWrap(True)
        self.output_summary.setStyleSheet("color: #8fa5bb;")
        output_form.addRow(self.output_summary)
        self.output_mode.currentIndexChanged.connect(self.output_settings_changed)
        self.frame_count.valueChanged.connect(self.output_settings_changed)
        _, executable_form = self.form_group("Simulator", simulation_layout)
        self.executable = QtWidgets.QLineEdit(str(ROOT / "simulation_release"))
        self.field_row(executable_form, "Executable", self.executable, "Compiled simulator launched locally for preview and integration. Changing it requires a new preview.")
        self.executable.textChanged.connect(self.invalidate_preview)
        choose = QtWidgets.QPushButton("Choose executable…")
        choose.clicked.connect(self.choose_executable)
        executable_form.addRow(choose)
        order = [self.simulation_fields["steps"], self.simulation_fields["dt"], self.output_mode,
                 self.frame_count, self.simulation_fields["snapshot_interval"],
                 self.simulation_fields["threads"], self.simulation_fields["theta"],
                 self.simulation_fields["softening"], self.executable, choose]
        for previous, following in zip(order, order[1:]):
            QtWidgets.QWidget.setTabOrder(previous, following)
        simulation_layout.addWidget(QtWidgets.QLabel("Fixed timestep · G = 1 · open boundaries"))
        simulation_layout.addStretch()
        for contents, label in ((object_page, "Structure"), (motion_page, "Motion"), (simulation_page, "Run settings")):
            scroll = QtWidgets.QScrollArea()
            scroll.setFrameShape(QtWidgets.QFrame.Shape.NoFrame)
            scroll.setWidgetResizable(True)
            scroll.setHorizontalScrollBarPolicy(QtCore.Qt.ScrollBarPolicy.ScrollBarAlwaysOff)
            scroll.setWidget(contents)
            editor_tabs.addTab(scroll, label)
        side.addWidget(editor_tabs, 1)
        self.total_label = QtWidgets.QLabel()
        side.addWidget(self.total_label)
        self.preview_button = QtWidgets.QPushButton("Generate preview")
        self.preview_button.setToolTip("Generate the exact initial state with the C++ simulator. Preview it before enabling Run.")
        self.preview_button.clicked.connect(self.generate_preview)
        side.addWidget(self.preview_button)
        self.run_button = QtWidgets.QPushButton("Run simulation")
        self.run_button.setObjectName("primaryAction")
        self.run_button.setToolTip("Run the currently previewed particle state with the selected run settings. Requires a current preview.")
        self.run_button.clicked.connect(self.start_simulation)
        side.addWidget(self.run_button)
        self.preview_notice = QtWidgets.QLabel("Generate a preview before starting a simulation.")
        self.preview_notice.setWordWrap(True)
        side.addWidget(self.preview_notice)
        self.vectors = QtWidgets.QCheckBox("Velocity vectors · sampled · × 0.05 T0")
        self.vectors.toggled.connect(self.toggle_vectors)
        side.addWidget(self.vectors)
        self.sections.addWidget(page)
        self.name.textEdited.connect(self.edit_object)
        self.model.currentIndexChanged.connect(self.model_changed)
        self.motion.currentIndexChanged.connect(self.edit_object)
        self.motion.currentIndexChanged.connect(self.update_motion_description)
        self.update_output_controls()

    def build_simulation(self):
        page = QtWidgets.QWidget()
        layout = QtWidgets.QVBoxLayout(page)
        form = QtWidgets.QFormLayout()
        self.integrator = QtWidgets.QComboBox()
        self.integrator.addItem("Explicit Euler", "euler")
        self.integrator.addItem("Leapfrog", "leapfrog")
        self.field_row(form, "Integrator", self.integrator,
            "Time integration method. Leapfrog uses second-order kick–drift–kick updates. Both use the selected fixed timestep; changing this preserves the prepared initial conditions.")
        self.output_root = QtWidgets.QLineEdit(str(ROOT / "output"))
        self.run_name = QtWidgets.QLineEdit(self.new_run_name())
        form.addRow("Run directory name", self.run_name)
        form.addRow("Output parent", self.output_root)
        layout.addLayout(form)
        row = QtWidgets.QVBoxLayout()
        self.button(row, "Choose output parent…", self.choose_output)
        self.cancel_button = self.button(row, "Cancel current operation", self.cancel_operation)
        layout.addLayout(row)
        self.progress = QtWidgets.QProgressBar()
        self.progress.setRange(0, 100)
        self.progress.setValue(0)
        layout.addWidget(self.progress)
        self.progress_label = QtWidgets.QLabel("Ready. Prepare an initial-condition preview in Setup.")
        self.progress_label.setWordWrap(True)
        layout.addWidget(self.progress_label)
        self.log = QtWidgets.QPlainTextEdit()
        self.log.setReadOnly(True)
        self.log.setMaximumBlockCount(5000)
        layout.addWidget(self.log, 1)
        self.sections.addWidget(page)

    def build_runs(self):
        page = QtWidgets.QWidget()
        layout = QtWidgets.QVBoxLayout(page)
        row = QtWidgets.QHBoxLayout()
        self.runs_root = QtWidgets.QLineEdit(str(ROOT / "output"))
        row.addWidget(self.runs_root, 1)
        self.button(row, "Browse…", self.choose_runs)
        self.button(row, "Refresh", self.refresh_runs)
        layout.addLayout(row)
        self.run_table = QtWidgets.QTableWidget(0, 4)
        self.run_table.setHorizontalHeaderLabels(["Run", "Status", "Particles", "Frames"])
        self.run_table.setSelectionBehavior(QtWidgets.QAbstractItemView.SelectionBehavior.SelectRows)
        self.run_table.setSelectionMode(QtWidgets.QAbstractItemView.SelectionMode.SingleSelection)
        self.run_table.setEditTriggers(QtWidgets.QAbstractItemView.EditTrigger.NoEditTriggers)
        self.run_table.horizontalHeader().setSectionResizeMode(0, QtWidgets.QHeaderView.ResizeMode.Stretch)
        for column in (1, 2, 3):
            self.run_table.horizontalHeader().setSectionResizeMode(column, QtWidgets.QHeaderView.ResizeMode.ResizeToContents)
        self.run_table.setWordWrap(False)
        self.run_table.itemSelectionChanged.connect(self.select_run)
        layout.addWidget(self.run_table, 1)
        row = QtWidgets.QVBoxLayout()
        self.open_run_button = self.button(row, "Open selected run", self.open_run)
        self.use_setup_button = self.button(row, "Use its setup", self.use_setup)
        self.delete_run_button = self.button(row, "Delete selected run…", self.delete_run)
        self.delete_run_button.setToolTip("Permanently delete the selected run directory, including its snapshots, setup and logs")
        self.run_summary = QtWidgets.QLabel("Select a saved run.")
        self.run_summary.setWordWrap(True)
        row.addWidget(self.run_summary)
        layout.addLayout(row)
        self.sections.addWidget(page)

    @staticmethod
    def new_run_name():
        return datetime.now().strftime("run_%Y%m%d_%H%M%S")

    def configuration(self):
        simulation = {key: widget.value() for key, widget in self.simulation_fields.items()}
        simulation["integrator"] = self.integrator.currentData()
        if self.output_mode.currentData() == "frames":
            simulation["frames"] = self.frame_count.value()
        return {"version": 1, "objects": copy.deepcopy(self.objects), "simulation": simulation}

    def signature(self):
        # Duration/output sampling cannot change the prepared particle state.
        initialization = {key: self.simulation_fields[key].value() for key in ("threads", "theta", "softening")}
        return json.dumps([self.objects, initialization, self.executable.text()], sort_keys=True)

    def update_output_controls(self):
        steps = self.simulation_fields["steps"].value()
        by_count = self.output_mode.currentData() == "frames"
        blocker = QtCore.QSignalBlocker(self.frame_count)
        self.frame_count.setRange(1 if steps == 0 else 2, steps + 1)
        del blocker
        self.frame_count.setVisible(by_count)
        self.frame_count_label.setVisible(by_count)
        self.simulation_fields["snapshot_interval"].setVisible(not by_count)
        self.simulation_labels["snapshot_interval"].setVisible(not by_count)
        interval = self.simulation_fields["snapshot_interval"].value()
        frames = self.frame_count.value() if by_count else (steps + interval - 1) // interval + 1
        duration = steps * self.simulation_fields["dt"].value()
        endpoints = "Initial state only." if steps == 0 else "Initial and final states included."
        self.output_summary.setText(f"{frames:,} saved frames · simulated duration {duration:g} T0\n{endpoints}")

    def output_settings_changed(self, *_):
        if self.editing: return
        self.update_output_controls()
        self.update_buttons()

    def invalidate_preview(self, *_):
        if self.editing: return
        self.preview_signature = None
        self.preview_notice.setText("Preview is out of date. Generate it again before Run.")
        count = sum(obj["particles"] for obj in self.objects)
        mass = sum(obj["mass"] for obj in self.objects)
        noun = "object" if len(self.objects) == 1 else "objects"
        self.total_label.setText(f"{len(self.objects)} {noun} · {count:,} particles · mass {mass:g} M0")
        self.update_buttons()

    def load_editor(self, index):
        if not 0 <= index < len(self.objects): return
        self.editing = True
        obj = self.objects[index]
        self.name.setText(obj["name"])
        self.model.setCurrentIndex(self.model.findData(obj["model"]))
        self.motion.setCurrentIndex(self.motion.findData(obj["velocity_mode"]))
        for key, widget in self.fields.items(): widget.setValue(obj[key])
        for key, widgets in self.transform_fields.items():
            for widget, value in zip(widgets, obj[key]): widget.setValue(value)
        self.editing = False
        self.update_model_fields()

    def update_model_fields(self):
        model = self.model.currentData()
        self.description.setText(DESCRIPTIONS[model])
        galaxy = model == "galaxy"
        self.galaxy_group.setVisible(galaxy)
        label, tooltip = SCALE_TIPS[model]
        self.field_labels["scale"].setText(label)
        self.field_labels["scale"].setToolTip(f"<qt>{html.escape(tooltip)}</qt>")
        self.fields["scale"].setToolTip(f"<qt>{html.escape(tooltip)}</qt>")
        self.fields["scale"].setAccessibleName(label)
        for key in ("height", "bulge_radius", "bulge_fraction", "q"):
            self.fields[key].setEnabled(galaxy)
        self.motion.setEnabled(not galaxy)
        self.motion.model().item(0).setEnabled(galaxy)
        self.update_motion_description()

    def update_motion_description(self, *_):
        self.motion_description.setText(MOTION_TIPS.get(self.motion.currentData(), ""))

    def model_changed(self, *_):
        if self.editing: return
        self.editing = True
        self.motion.setCurrentIndex(self.motion.findData("equilibrium" if self.model.currentData() == "galaxy" else "random"))
        self.editing = False
        self.update_model_fields()
        self.edit_object()

    def edit_object(self, *_):
        index = self.object_list.currentRow()
        if self.editing or not 0 <= index < len(self.objects): return
        obj = self.objects[index]
        obj.update({key: widget.value() for key, widget in self.fields.items()})
        obj.update({key: [widget.value() for widget in widgets] for key, widgets in self.transform_fields.items()})
        obj.update(name=self.name.text(), model=self.model.currentData(), velocity_mode=self.motion.currentData())
        self.object_list.item(index).setText(obj["name"] or "Unnamed object")
        self.invalidate_preview()

    def add_object(self):
        obj = default_object(len(self.objects) + 1)
        obj["position"][0] = 2.0 * len(self.objects)
        self.objects.append(obj)
        self.object_list.addItem(obj["name"])
        self.object_list.setCurrentRow(len(self.objects) - 1)
        self.invalidate_preview()

    def duplicate_object(self):
        obj = copy.deepcopy(self.objects[self.object_list.currentRow()])
        obj["name"] += " copy"
        obj["seed"] = (obj["seed"] + 1) % 2_147_483_648
        obj["position"][0] += 2.0
        self.objects.append(obj)
        self.object_list.addItem(obj["name"])
        self.object_list.setCurrentRow(len(self.objects) - 1)
        self.invalidate_preview()

    def remove_object(self):
        if len(self.objects) == 1: return
        index = self.object_list.currentRow()
        del self.objects[index]
        self.object_list.takeItem(index)
        self.object_list.setCurrentRow(min(index, len(self.objects) - 1))
        self.invalidate_preview()

    def update_buttons(self):
        if not hasattr(self, "run_button"): return
        busy = self.process is not None or self.loading_future is not None
        self.preview_button.setEnabled(not busy)
        preview_ready = self.preview_view is not None and self.preview_view.view.isValid()
        self.run_button.setEnabled(not busy and preview_ready and self.preview_signature is not None and self.preview_signature == self.signature())
        self.remove_button.setEnabled(len(self.objects) > 1)
        if hasattr(self, "cancel_button"): self.cancel_button.setEnabled(self.process is not None)
        if hasattr(self, "open_run_button"):
            path = self.selected_run()
            self.open_run_button.setEnabled(not busy and path is not None and self.run_is_complete(path))
            self.use_setup_button.setEnabled(not busy and path is not None and (path / "configuration.json").exists())
            self.delete_run_button.setEnabled(not busy and path is not None)

    def executable_path(self):
        path = Path(self.executable.text()).expanduser().resolve()
        if not path.is_file() or not os.access(path, os.X_OK):
            raise ValueError("Choose a compiled simulator executable. Build simulation_release before launching a run.")
        return path

    def generate_preview(self):
        if self.process is not None or self.loading_future is not None: return
        try:
            executable = self.executable_path()
            if sum(obj["particles"] for obj in self.objects) > 2_147_483_647:
                raise ValueError("Total particle count exceeds the simulator's 32-bit index limit")
            self.close_view("preview")
            if self.preview_temp is not None: self.preview_temp.cleanup()
            self.preview_temp = tempfile.TemporaryDirectory(prefix="barnes-hut-preview-")
            temp = Path(self.preview_temp.name)
            self.preparing_config = self.configuration()
            self.preparing_signature = self.signature()
            self.preview_signature = None
            write_scene(temp / "scene.txt", self.preparing_config["objects"])
            self.preview_notice.setText("Preparing initial conditions with the C++ simulator…")
            args = simulation_arguments(self.preparing_config) + ["--scene", str(temp / "scene.txt"),
                    "--prepare-only", "--output", str(temp / "prepared")]
            self.start_process(executable, args, "preview")
        except (OSError, ValueError) as error:
            self.show_error(error)

    def start_simulation(self):
        if self.process is not None or self.loading_future is not None or self.preview_signature is None or self.preview_signature != self.signature(): return
        try:
            executable = self.executable_path()
            configuration = self.configuration()
            name = self.run_name.text().strip()
            if not name or name in (".", "..") or Path(name).name != name or "/" in name or "\\" in name:
                raise ValueError("Run name must be a single directory name")
            parent = Path(self.output_root.text()).expanduser().resolve()
            parent.mkdir(parents=True, exist_ok=True)
            path = parent / name
            path.mkdir()  # Exclusive: never replace an existing run.
            self.current_run = path
            self.last_step = 0
            temp = Path(self.preview_temp.name)
            shutil.copyfile(temp / "scene.txt", path / "scene.txt")
            shutil.copyfile(temp / "prepared" / "initial_state.bin", path / "initial_state.bin")
            shutil.copyfile(temp / "prepared" / "metadata.json", path / "initial_metadata.json")
            (path / "configuration.json").write_text(json.dumps(configuration, indent=2) + "\n")
            self.log_file = (path / "simulation.log").open("w", encoding="utf-8")
            self.write_status("running")
            args = simulation_arguments(configuration) + ["--scene", str(path / "scene.txt"),
                    "--initial-state", str(path / "initial_state.bin"), "--output", str(path / "snapshots")]
            self.runs_root.setText(str(parent))
            self.sections.setCurrentIndex(1)
            self.start_process(executable, args, "simulation")
        except (OSError, ValueError) as error:
            self.show_error(error)

    def start_process(self, executable, arguments, operation):
        process = QtCore.QProcess(self)
        process.setProgram(str(executable))
        process.setArguments(arguments)
        process.setWorkingDirectory(str(ROOT))
        self.process, self.operation = process, operation
        self.cancelled = False
        self.stdout_buffer = ""
        self.started_at = time.perf_counter()
        self.last_step = 0
        self.log.clear()
        self.progress.setRange(0, 0)
        self.progress_label.setText("Preparing initial conditions…" if operation == "preview" else "Starting simulation…")
        process.readyReadStandardOutput.connect(self.read_stdout)
        process.readyReadStandardError.connect(self.read_stderr)
        process.finished.connect(self.process_finished)
        process.errorOccurred.connect(self.process_error)
        process.start()
        self.update_buttons()

    def append_log(self, text):
        if not text: return
        self.log.moveCursor(QtGui.QTextCursor.MoveOperation.End)
        self.log.insertPlainText(text)
        if self.log_file is not None:
            try:
                self.log_file.write(text)
                self.log_file.flush()
            except OSError as error:
                self.statusBar().showMessage(f"Could not write simulation log: {error}")

    def read_stdout(self):
        if self.process is None: return
        text = bytes(self.process.readAllStandardOutput()).decode("utf-8", errors="replace")
        self.append_log(text)
        self.stdout_buffer += text
        while "\n" in self.stdout_buffer:
            line, self.stdout_buffer = self.stdout_buffer.split("\n", 1)
            match = re.match(r"Step (\d+)/(\d+), time (.+)", line)
            if match:
                step, total = int(match[1]), int(match[2])
                self.last_step = step
                self.progress.setRange(0, 1000)
                self.progress.setValue(round(1000 * step / max(total, 1)))
                elapsed = time.perf_counter() - self.started_at
                remaining = elapsed * (total - step) / max(step, 1)
                self.progress_label.setText(f"Step {step:,} / {total:,} · t = {match[3]} T0 · elapsed {elapsed:.1f}s · remaining ≈ {remaining:.1f}s")

    def read_stderr(self):
        if self.process is not None:
            self.append_log(bytes(self.process.readAllStandardError()).decode("utf-8", errors="replace"))

    def process_error(self, error):
        if error == QtCore.QProcess.ProcessError.FailedToStart:
            self.append_log(self.process.errorString() + "\n")
            self.process_finished(-1, QtCore.QProcess.ExitStatus.CrashExit)

    def process_finished(self, code, exit_status):
        if self.process is None: return
        self.read_stdout()
        self.read_stderr()
        operation = self.operation
        process, self.process = self.process, None
        process.deleteLater()
        success = code == 0 and exit_status == QtCore.QProcess.ExitStatus.NormalExit and not self.cancelled
        self.progress.setRange(0, 1000)
        self.progress.setValue(1000 if success else 0)
        result = "Completed" if success else ("Cancelled" if self.cancelled else "Failed")
        elapsed = time.perf_counter() - self.started_at
        self.progress_label.setText(f"{result} · elapsed {elapsed:.1f}s")
        if operation == "simulation":
            self.write_status(result.lower())
            if self.log_file is not None:
                self.log_file.close()
                self.log_file = None
            self.run_name.setText(self.new_run_name())
            self.refresh_runs(self.current_run)
            if success:
                self.statusBar().showMessage("Simulation completed. Select it in Saved Runs to open playback.", 10000)
        elif success and self.preparing_signature == self.signature():
            self.begin_load(Path(self.preview_temp.name) / "prepared", "preview")
        else:
            self.preview_notice.setText("Preview cancelled or failed." if not success else "Settings changed during preparation. Generate a new preview.")
        self.operation = None
        self.update_buttons()
        if not success and not self.cancelled and not self.closing:
            details = f"See {self.current_run / 'simulation.log'}" if operation == "simulation" else "See the Simulation section for details."
            self.show_error(f"{'Simulation' if operation == 'simulation' else 'Initial-condition preparation'} "
                            f"failed (exit code {code}).\n{details}\nCheck the settings and simulator output before retrying.")
        if self.closing: self.close()

    def cancel_operation(self):
        if self.process is None: return
        self.cancelled = True
        process = self.process
        process.terminate()
        self.progress_label.setText("Cancelling… Partial run files will be retained.")
        QtCore.QTimer.singleShot(3000, lambda: process.kill() if self.process is process else None)

    def write_status(self, status):
        try:
            (self.current_run / "status.json").write_text(json.dumps({"status": status,
                "updated": datetime.now().astimezone().isoformat(), "completed_step": self.last_step if hasattr(self, "last_step") else 0}, indent=2) + "\n")
        except OSError as error:
            self.statusBar().showMessage(f"Could not write run status: {error}")

    def begin_load(self, path, kind):
        if self.loading_future is not None: return
        self.loading_path, self.loading_kind = path, kind
        self.loading_future = self.loader.submit(load_display, path, kind == "preview")
        self.statusBar().showMessage("Loading and validating snapshots…")
        self.update_buttons()

    def poll(self):
        if self.loading_future is None or not self.loading_future.done(): return
        future, self.loading_future = self.loading_future, None
        try:
            frames, metadata, bounds = future.result()
            kind = self.loading_kind
            if kind == "preview" and self.preparing_signature != self.signature():
                self.preview_notice.setText("Settings changed while loading. Generate a new preview.")
                self.update_buttons()
                return
            widget = create_viewer(frames, name=self.loading_path.name, metadata=metadata, bounds=bounds)
            self.close_view(kind)
            if kind == "preview":
                self.preview_view = widget
                widget.view.renderer_ready.connect(self.update_buttons)
                self.vector_item = gl.GLLinePlotItem(pos=velocity_vectors(self.loading_path / "initial_state.bin"),
                    color=(.4, .85, .95, .6), mode="lines", glOptions="translucent")
                widget.view.addItem(self.vector_item)
                self.toggle_vectors(self.vectors.isChecked())
                self.preview_signature = self.preparing_signature
                self.preview_notice.setText("Prepared state is current. Run will use these exact positions, velocities and masses.")
                self.sections.setCurrentIndex(0)
            else:
                self.run_view = widget
                self.run_view_path = self.loading_path.resolve()
                self.sections.setCurrentIndex(2)
            previous = self.display.currentWidget()
            if previous is not self.preview_placeholder and previous.playback.playing:
                previous.toggle_playback()
            self.display.addWidget(widget)
            self.display.setCurrentWidget(widget)
            self.statusBar().clearMessage()
        except (OSError, ValueError, RuntimeError, EOFError) as error:
            kind = self.loading_kind
            if kind == "preview":
                self.preview_signature = None
                self.preview_notice.setText("Preview could not be loaded. Generate a new preview.")
            self.show_error(f"Could not load {'the preview' if kind == 'preview' else 'the selected run'}:\n{error}")
            if kind != "preview": self.refresh_runs(self.loading_path)
        self.update_buttons()

    def close_view(self, kind):
        attribute = "preview_view" if kind == "preview" else "run_view"
        widget = getattr(self, attribute)
        if widget is not None:
            widget.close()
            self.display.removeWidget(widget)
            widget.deleteLater()
            setattr(self, attribute, None)
        if kind == "preview": self.vector_item = None
        else: self.run_view_path = None

    def toggle_vectors(self, checked):
        if self.vector_item is not None: self.vector_item.setVisible(checked)

    def selected_run(self):
        if not hasattr(self, "run_table") or self.run_table.currentRow() < 0: return None
        item = self.run_table.item(self.run_table.currentRow(), 0)
        return Path(item.data(QtCore.Qt.ItemDataRole.UserRole)) if item is not None else None

    def run_is_complete(self, path):
        if path == self.selected_run():
            status_item = self.run_table.item(self.run_table.currentRow(), 1)
            if status_item is not None and status_item.text() != "Completed": return False
        status_file = path / "status.json"
        if status_file.exists():
            try:
                if read_json(status_file)["status"] != "completed": return False
            except (OSError, ValueError, KeyError): return False
        return (snapshot_directory(path) / "metadata.json").exists()

    def refresh_runs(self, selected=None):
        selected = selected or self.selected_run()
        root = Path(self.runs_root.text()).expanduser()
        self.run_table.setRowCount(0)
        try:
            directories = sorted((p for p in root.iterdir() if p.is_dir()), key=lambda p: p.stat().st_mtime, reverse=True)
            for path in directories:
                directory = snapshot_directory(path)
                meta_path, status_path = directory / "metadata.json", path / "status.json"
                if not meta_path.exists() and not status_path.exists() and not (path / "configuration.json").exists(): continue
                problem = ""
                try:
                    metadata = read_json(meta_path) if meta_path.exists() else {}
                    status = read_json(status_path).get("status", "incomplete") if status_path.exists() else "completed"
                    if not isinstance(status, str): raise ValueError("Invalid run status")
                    if status == "running" and (self.process is None or path != self.current_run): status = "interrupted"
                    count = metadata.get("particle_count", "—")
                    if count == "—" and (path / "configuration.json").exists():
                        count = sum(obj["particles"] for obj in read_json(path / "configuration.json")["objects"])
                    frames = len(metadata.get("snapshots", []))
                    if status == "completed":
                        try:
                            if not meta_path.exists(): raise FileNotFoundError("Run metadata.json is missing. Restore it or choose another run.")
                            directory_frames(directory, metadata)
                        except (OSError, ValueError) as error:
                            status, problem = "damaged", str(error)
                except (OSError, ValueError, KeyError, TypeError) as error:
                    status, count, frames = "unreadable", "—", "—"
                    problem = str(error)
                row = self.run_table.rowCount()
                self.run_table.insertRow(row)
                for column, value in enumerate((path.name, status.title(), f"{count:,}" if isinstance(count, int) else count, frames)):
                    self.run_table.setItem(row, column, QtWidgets.QTableWidgetItem(str(value)))
                self.run_table.item(row, 0).setData(QtCore.Qt.ItemDataRole.UserRole, str(path.resolve()))
                self.run_table.item(row, 0).setData(QtCore.Qt.ItemDataRole.UserRole + 1, problem)
                for column in range(4): self.run_table.item(row, column).setToolTip(problem)
                if selected is not None and path.resolve() == selected.resolve(): self.run_table.selectRow(row)
        except OSError as error:
            self.statusBar().showMessage(str(error))
        self.update_buttons()

    def select_run(self):
        path = self.selected_run()
        if path is not None:
            problem = self.run_table.item(self.run_table.currentRow(), 0).data(QtCore.Qt.ItemDataRole.UserRole + 1)
            if problem:
                self.run_summary.setText(problem)
                self.run_summary.setWordWrap(True)
                self.update_buttons()
                return
            try:
                file = snapshot_directory(path) / "metadata.json"
                metadata = read_json(file) if file.exists() else {}
                self.run_summary.setText(f"{metadata.get('step_count', '—')} steps · dt {metadata.get('timestep', '—')} · {metadata.get('thread_count', '—')} threads")
            except (OSError, ValueError): self.run_summary.setText("Could not read metadata")
        else:
            self.run_summary.setText("Select a saved run.")
        self.update_buttons()

    def delete_run(self):
        if self.process is not None or self.loading_future is not None: return
        path = self.selected_run()
        if path is None: return
        try:
            root = Path(self.runs_root.text()).expanduser().resolve()
            entry = root / self.run_table.item(self.run_table.currentRow(), 0).text()
            if entry.is_symlink() or entry.resolve() != path or path.parent != root:
                raise ValueError("Only run directories directly inside the selected runs folder can be deleted. Symbolic links cannot be deleted here.")
            if not path.is_dir():
                raise FileNotFoundError(f"The run directory no longer exists: {path}")
            dialog = QtWidgets.QMessageBox(QtWidgets.QMessageBox.Icon.Warning,
                "Delete saved run", f"Permanently delete this run?\n\n{path}\n\n"
                "All files in this directory, including snapshots, setup and logs, will be deleted. This cannot be undone.",
                QtWidgets.QMessageBox.StandardButton.NoButton, self)
            delete_button = dialog.addButton("Delete run", QtWidgets.QMessageBox.ButtonRole.DestructiveRole)
            cancel_button = dialog.addButton(QtWidgets.QMessageBox.StandardButton.Cancel)
            dialog.setDefaultButton(cancel_button)
            dialog.setEscapeButton(cancel_button)
            dialog.exec()
            if dialog.clickedButton() is not delete_button: return
            # Check again after the modal dialog in case the directory changed.
            if entry.is_symlink() or entry.resolve() != path or not path.is_dir():
                raise ValueError("The run directory changed while confirming deletion. Refresh the list and try again.")
            if self.run_view_path == path:
                self.close_view("run")
            shutil.rmtree(path)
            self.statusBar().showMessage(f"Deleted run: {path.name}", 10000)
        except (OSError, ValueError) as error:
            self.show_error(f"Could not delete the run:\n{error}")
        self.refresh_runs()

    def open_run(self):
        path = self.selected_run()
        if path is None: return
        if not self.run_is_complete(path):
            self.show_error("The selected run is missing or is not complete. Refresh the list and choose a completed run.")
            self.refresh_runs(path)
            return
        self.begin_load(path, "run")

    def use_setup(self):
        path = self.selected_run()
        try:
            config = read_json(path / "configuration.json")
            if config.get("version") != 1 or not config.get("objects"): raise ValueError("Unsupported setup configuration")
            self.editing = True
            self.object_list.clear()
            self.objects = config["objects"]
            for obj in self.objects: self.object_list.addItem(obj["name"])
            for key, widget in self.simulation_fields.items(): widget.setValue(config["simulation"][key])
            integrator_index = self.integrator.findData(config["simulation"].get("integrator", "euler"))
            if integrator_index < 0: raise ValueError("Unsupported saved integrator")
            self.integrator.setCurrentIndex(integrator_index)
            frames = config["simulation"].get("frames")
            self.output_mode.setCurrentIndex(0 if frames is not None else 1)
            self.update_output_controls()
            if frames is None:
                steps, interval = config["simulation"]["steps"], config["simulation"]["snapshot_interval"]
                frames = (steps + interval - 1) // interval + 1
            self.frame_count.setValue(frames)
            self.update_output_controls()
            self.object_list.setCurrentRow(0)
            self.load_editor(0)
            self.editing = False
            self.invalidate_preview()
            self.sections.setCurrentIndex(0)
        except (OSError, ValueError, KeyError, TypeError) as error:
            self.editing = False
            self.show_error(error)

    def choose_executable(self):
        path, _ = QtWidgets.QFileDialog.getOpenFileName(self, "Simulator executable", str(ROOT))
        if path: self.executable.setText(path)

    def choose_output(self):
        path = QtWidgets.QFileDialog.getExistingDirectory(self, "Output parent", self.output_root.text())
        if path: self.output_root.setText(path)

    def choose_runs(self):
        path = QtWidgets.QFileDialog.getExistingDirectory(self, "Saved runs parent", self.runs_root.text())
        if path:
            self.runs_root.setText(path)
            self.refresh_runs()

    def show_error(self, error):
        self.statusBar().showMessage(str(error), 15000)
        QtWidgets.QMessageBox.warning(self, "Simulation studio", str(error))
        self.update_buttons()

    def changeEvent(self, event):
        super().changeEvent(event)
        if (event.type() == QtCore.QEvent.Type.WindowStateChange
                and self.isMaximized() and not self.isFullScreen()):
            # Let the native maximize transition finish before filling the screen.
            QtCore.QTimer.singleShot(0, lambda: self.showFullScreen() if self.isMaximized() else None)

    def closeEvent(self, event):
        if self.process is not None:
            self.closing = True
            self.cancel_operation()
            event.ignore()
            return
        self.poll_timer.stop()
        self.close_view("preview")
        self.close_view("run")
        self.loader.shutdown(wait=False, cancel_futures=True)
        if self.log_file is not None: self.log_file.close()
        if self.preview_temp is not None: self.preview_temp.cleanup()
        event.accept()


def main():
    QtCore.QCoreApplication.setAttribute(QtCore.Qt.ApplicationAttribute.AA_ShareOpenGLContexts)
    pg.setConfigOptions(background=BACKGROUND, foreground="#b7c9db", antialias=False)
    app = QtWidgets.QApplication(sys.argv)
    window = SimulationApp()
    window.show()
    sys.exit(app.exec())


if __name__ == "__main__":
    main()
