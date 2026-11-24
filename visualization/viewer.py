"""Play NumPy particle snapshots in a native Qt/OpenGL viewer."""

import argparse
from concurrent.futures import ThreadPoolExecutor
import json
import math
from pathlib import Path
import sys
import time

import numpy as np
from OpenGL import GL
from PySide6 import QtCore, QtGui, QtWidgets
import pyqtgraph as pg
import pyqtgraph.opengl as gl

BACKGROUND = "#0b111a"
COLORMAPS = ("inferno", "magma", "viridis", "plasma", "cividis", "gray")
VIEWS = {"Home": (24, -55), "XY": (90, -90), "XZ": (0, -90), "YZ": (0, 0)}


def directory_frames(source, metadata):
    """Use recorded frame order and report deleted snapshots rather than skip them."""
    if not isinstance(metadata, dict):
        raise ValueError("metadata.json must contain a JSON object")
    if "snapshots" in metadata:
        snapshots = metadata["snapshots"]
        if not isinstance(snapshots, list) or not snapshots:
            raise ValueError("metadata.json contains no valid snapshot list")
        frames = []
        for snapshot in snapshots:
            if not isinstance(snapshot, dict):
                raise ValueError("Invalid snapshot entry in metadata.json")
            name = snapshot.get("file")
            if not isinstance(name, str) or not name or Path(name).name != name or not name.endswith(".npy"):
                raise ValueError("Invalid snapshot filename in metadata.json")
            if "time" in snapshot and (not isinstance(snapshot["time"], (int, float)) or
                                       not np.isfinite(snapshot["time"])):
                raise ValueError(f"Invalid snapshot time for {name}")
            frames.append(source / name)
        if len(set(frames)) != len(frames):
            raise ValueError("metadata.json lists duplicate snapshots")
    else:
        frames = sorted(source.glob("frame_*.npy"))
    if not frames:
        raise ValueError(f"No snapshots found in {source}")
    missing = [frame.name for frame in frames if not frame.is_file()]
    if missing:
        examples = ", ".join(missing[:3]) + (", …" if len(missing) > 3 else "")
        raise FileNotFoundError(f"{len(missing)} snapshot(s) missing from {source}: {examples}. "
                                "Restore the files or select another run.")
    return frames


def load_frames(source, metadata=None):
    """Read a stacked demo (F, N, 3), a single frame, or a frame directory."""
    if source.is_dir():
        if metadata is None:
            path = source / "metadata.json"
            metadata = json.loads(path.read_text()) if path.exists() else {}
        return directory_frames(source, metadata)
    frames = np.load(source, mmap_mode="r", allow_pickle=False)
    if not isinstance(frames, np.ndarray):
        if hasattr(frames, "close"): frames.close()
        raise ValueError("Expected a NumPy .npy array")
    if frames.ndim == 2:
        frames = frames[np.newaxis, ...]
    if frames.ndim != 3 or len(frames) == 0:
        raise ValueError("Expected a snapshot (N, 3) or an animation (frames, N, 3)")
    return frames


def read_frame(frames, index):
    item = frames[index]
    positions = np.load(item, mmap_mode="r", allow_pickle=False) if isinstance(item, Path) else item
    if not isinstance(positions, np.ndarray):
        if hasattr(positions, "close"): positions.close()
        raise ValueError("Expected a NumPy .npy particle array")
    return positions


def validate_frame(positions, count=None):
    if positions.ndim != 2 or positions.shape[1] != 3 or not len(positions):
        raise ValueError("Every frame must have a nonempty (N, 3) shape")
    if count is not None and len(positions) != count:
        raise ValueError(f"Expected {count:,} particles, found {len(positions):,}")
    if not np.issubdtype(positions.dtype, np.number) or np.iscomplexobj(positions):
        raise ValueError("Positions must contain real numbers")
    if not np.isfinite(positions).all():
        raise ValueError("Positions must be finite")
    minimum, maximum = positions.min(axis=0), positions.max(axis=0)
    if np.max(maximum) > np.finfo(np.float32).max or np.min(minimum) < -np.finfo(np.float32).max:
        raise ValueError("Positions exceed the float32 rendering range")
    return minimum, maximum


def frame_bounds(frames):
    """Validate and scan once for global bounds, without retaining snapshots."""
    minimum = np.full(3, np.inf)
    maximum = np.full(3, -np.inf)
    count = None
    for index in range(len(frames)):
        try:
            positions = read_frame(frames, index)
            frame_min, frame_max = validate_frame(positions, count)
        except (OSError, ValueError, EOFError) as error:
            source = frames[index]
            name = str(source) if isinstance(source, Path) else f"frame {index + 1}"
            raise ValueError(f"Cannot load {name}: {error}") from error
        if count is None:
            count = len(positions)
        minimum = np.minimum(minimum, frame_min)
        maximum = np.maximum(maximum, frame_max)
    center = minimum / 2 + maximum / 2
    half_width = max(float(np.max(maximum - minimum)) / 2, 1e-6) * 1.15
    return count, center, half_width


class Playback:
    """Wall-clock playback state, independent of rendering and the camera."""

    def __init__(self, count, fps):
        self.count, self.fps = count, fps
        self.position, self.speed = 0.0, 1.0
        self.playing = count > 1
        self.last_tick = time.perf_counter()

    def seek(self, index):
        self.position = float(index)
        self.last_tick = time.perf_counter()

    def advance(self):
        now = time.perf_counter()
        elapsed, self.last_tick = now - self.last_tick, now
        if self.playing:
            self.position = (self.position + elapsed * self.fps * self.speed) % self.count
        return int(self.position)


class ParticleView(gl.GLViewWidget):
    """Orthographic camera; the same view transform drives the density panel."""

    camera_changed = QtCore.Signal()
    renderer_ready = QtCore.Signal(str)

    def __init__(self, center, half_width):
        super().__init__()
        self.global_center = np.asarray(center)
        self.global_radius = math.sqrt(3) * half_width
        self.setBackgroundColor(BACKGROUND)
        self.setMinimumSize(320, 240)
        self.orientation = QtWidgets.QLabel(self)
        self.scale_label = QtWidgets.QLabel(self)
        for label in (self.orientation, self.scale_label):
            label.setAttribute(QtCore.Qt.WidgetAttribute.WA_TransparentForMouseEvents)
            label.setStyleSheet("background: transparent; color: #7f93aa;")
        self.camera_changed.connect(self.update_indicators)

    def half_extents(self):
        half_x = self.opts["distance"] * math.tan(math.radians(self.opts["fov"]) / 2)
        return half_x, half_x * max(1, self.height()) / max(1, self.width())

    def projectionMatrix(self, region, viewport):
        x0, y0, width, height = viewport
        half_x, half_y = self.half_extents()
        left = half_x * (2 * (region[0] - x0) / width - 1)
        right = half_x * (2 * (region[0] + region[2] - x0) / width - 1)
        bottom = half_y * (2 * (region[1] - y0) / height - 1)
        top = half_y * (2 * (region[1] + region[3] - y0) / height - 1)
        # Orthographic near may be negative. Include the global bounds even
        # after focusing/zooming, so GL and density integrate the same depths.
        center = np.array(self.opts["center"].toTuple())
        radius = self.global_radius + np.linalg.norm(center - self.global_center)
        distance = self.opts["distance"]
        matrix = QtGui.QMatrix4x4()
        matrix.ortho(left, right, bottom, top, distance - radius, distance + radius)
        return matrix

    def density_camera(self):
        matrix = self.viewMatrix()
        # Explicit rows avoid ambiguity about Qt's column-major data() layout.
        rows = np.array([matrix.row(i).toTuple() for i in range(2)], dtype=np.float64)
        return rows, self.half_extents(), (self.width(), self.height())

    def initializeGL(self):
        super().initializeGL()
        from OpenGL.GL import GL_RENDERER, GL_VERSION, glGetString
        renderer = glGetString(GL_RENDERER).decode(errors="replace")
        version = glGetString(GL_VERSION).decode(errors="replace")
        software = any(name in renderer.lower() for name in ("llvmpipe", "softpipe", "software", "swrast"))
        status = f"OpenGL: {renderer} — {'SOFTWARE rendering' if software else 'check vendor GPU identity'}"
        print(f"{status}\nOpenGL version: {version}", flush=True)
        self.renderer_ready.emit(status)

    def update_indicators(self):
        # A fixed-size orientation triad remains readable after focusing on a
        # small core, unlike world-space axes sized from distant outliers.
        ratio = self.devicePixelRatioF()
        pixmap = QtGui.QPixmap(round(110 * ratio), round(110 * ratio))
        pixmap.setDevicePixelRatio(ratio)
        pixmap.fill(QtCore.Qt.GlobalColor.transparent)
        painter = QtGui.QPainter(pixmap)
        painter.setRenderHint(QtGui.QPainter.RenderHint.Antialiasing)
        painter.setFont(QtGui.QFont("Sans Serif", 9))
        matrix = self.viewMatrix()
        origin = QtCore.QPointF(55, 55)
        for index, (label, color) in enumerate((("X", "#bb7780"), ("Y", "#83b59a"), ("Z", "#839fc9"))):
            direction = QtCore.QPointF(matrix.row(0).toTuple()[index], -matrix.row(1).toTuple()[index])
            painter.setPen(QtGui.QPen(QtGui.QColor(color), 1.5))
            painter.drawLine(origin, origin + direction * 28)
            painter.drawText(origin + direction * 40 + QtCore.QPointF(-4, 4), label)
        painter.end()
        self.orientation.setPixmap(pixmap)
        self.orientation.resize(110, 110)
        self.orientation.move(0, max(0, self.height() - 110))
        half_x, half_y = self.half_extents()
        self.scale_label.setText(f"View: {2 * half_x:.3g} × {2 * half_y:.3g} L0")
        self.scale_label.adjustSize()
        self.scale_label.move(12, 10)

    def orbit(self, *args, **kwargs):
        super().orbit(*args, **kwargs)
        self.camera_changed.emit()

    def pan(self, dx, dy, dz, relative="global"):
        # Pan in the camera plane, including top-down presets.
        super().pan(dx, dy, dz, relative="view" if relative != "global" else "global")
        self.camera_changed.emit()

    def wheelEvent(self, event):
        delta = event.angleDelta().y() or event.angleDelta().x()
        self.opts["distance"] = float(np.clip(self.opts["distance"] * 0.999 ** delta, 1e-12, 1e30))
        self.update()
        self.camera_changed.emit()

    def setCameraPosition(self, *args, **kwargs):
        super().setCameraPosition(*args, **kwargs)
        self.camera_changed.emit()

    def resizeEvent(self, event):
        super().resizeEvent(event)
        self.camera_changed.emit()


def density_grid(half_extents, viewport, resolution=None):
    """Square world-space bins; Auto targets one bin per displayed device pixel."""
    longest = min(4096, max(viewport)) if resolution is None else resolution
    half_x, half_y = half_extents
    return (max(1, round(longest * half_x / max(half_x, half_y))),
            max(1, round(longest * half_y / max(half_x, half_y))))


def smoothing_boxes(sigma):
    """Three odd-width box filters approximate a Gaussian of the requested RMS."""
    lower = max(1, int(math.sqrt(4 * sigma * sigma + 1)))
    lower -= 1 - lower % 2
    count_lower = round((12 * sigma * sigma - 3 * lower * lower - 12 * lower - 9) /
                        (-4 * lower - 4))
    count_lower = max(0, min(3, count_lower))
    return [lower] * count_lower + [lower + 2] * (3 - count_lower)


def smooth_density(counts, boxes):
    # Cumulative-sum box filters keep large smoothing settings linear in image
    # size instead of making hundreds of full-image kernel passes.
    roundoff = np.finfo(counts.dtype).eps * counts.sum() * 8
    for axis, widths in enumerate(boxes):
        for width in widths:
            if width == 1:
                continue
            radius = width // 2
            padding = [(0, 0), (0, 0)]
            padding[axis] = (radius + 1, radius)
            integral = np.cumsum(np.pad(counts, padding), axis=axis)
            start, end = [slice(None)] * 2, [slice(None)] * 2
            start[axis], end[axis] = slice(width, None), slice(None, -width)
            counts = (integral[tuple(start)] - integral[tuple(end)]) / width
    # Prefix-sum cancellation must not create tiny positive "density" in empty
    # cells and distort positive-cell percentiles used by the explicit Fit action.
    counts[counts < roundoff] = 0
    return counts


def projected_density(positions, rows, half_extents, viewport, resolution=None, smoothing=1):
    """Number per projected simulation area, integrated along the view direction.

    Row zero is the bottom image row. No mass weighting or 3D density estimate
    is implied. Smoothing is an approximate Gaussian RMS in displayed device
    pixels. A padded histogram includes neighboring offscreen contributions.
    """
    half_x, half_y = half_extents
    columns, rows_count = density_grid(half_extents, viewport, resolution)
    boxes_x = smoothing_boxes(smoothing * columns / max(1, viewport[0]))
    boxes_y = smoothing_boxes(smoothing * rows_count / max(1, viewport[1]))
    margin_x = sum(width // 2 for width in boxes_x)
    margin_y = sum(width // 2 for width in boxes_y)
    cell_width, cell_height = 2 * half_x / columns, 2 * half_y / rows_count
    projected = positions @ rows[:, :3].T + rows[:, 3]
    counts, _, _ = np.histogram2d(projected[:, 1], projected[:, 0],
                                 bins=(rows_count + 2 * margin_y, columns + 2 * margin_x),
                                 range=((-half_y - margin_y * cell_height, half_y + margin_y * cell_height),
                                        (-half_x - margin_x * cell_width, half_x + margin_x * cell_width)))
    counts = smooth_density(counts, (boxes_y, boxes_x))
    visible = counts[margin_y:margin_y + rows_count, margin_x:margin_x + columns]
    return np.asarray(visible / (cell_width * cell_height), dtype=np.float32)


class SmoothImageItem(pg.ImageItem):
    def paint(self, painter, *args):
        painter.save()
        painter.setRenderHint(QtGui.QPainter.RenderHint.SmoothPixmapTransform)
        super().paint(painter, *args)
        painter.restore()


class DensityPanel(pg.GraphicsLayoutWidget):
    """Persistent image and log-density legend with independently held limits."""

    def __init__(self):
        super().__init__()
        self.setMinimumWidth(300)
        self.plot = self.addPlot(title="Projected number density · camera right / up")
        self.plot.setLabel("bottom", "Camera horizontal", units="L0")
        self.plot.setLabel("left", "Camera vertical", units="L0")
        self.plot.setAspectLocked(True)
        self.plot.setMouseEnabled(x=False, y=False)
        self.plot.hideButtons()
        self.plot.setMenuEnabled(False)
        self.image = SmoothImageItem(axisOrder="row-major")
        self.plot.addItem(self.image)
        self.legend = pg.ColorBarItem(values=(-2, 2), colorMap="inferno", interactive=False,
                                      label="log10(number / L0²)", colorMapMenu=False)
        self.legend.setImageItem(self.image, insert_in=self.plot)
        self.sampling_label = self.addLabel(row=1, col=0, justify="left", color="#7f93aa")
        self.density = None
        self.half_extents = None
        self.plot_only = False
        self.normal_margins = self.ci.layout.getContentsMargins()

    def set_plot_only(self, enabled):
        self.plot_only = enabled
        self.plot.getAxis("left").setVisible(not enabled)
        self.plot.getAxis("bottom").setVisible(not enabled)
        self.legend.setVisible(not enabled)
        self.sampling_label.setVisible(not enabled)
        title = self.plot.titleLabel.text
        self.plot.setTitle(None if enabled else title)
        self.ci.layout.setContentsMargins(*(0, 0, 0, 0) if enabled else self.normal_margins)

    def set_frame_title(self, text):
        if self.plot_only:
            self.plot.titleLabel.setText(text)
        else:
            self.plot.setTitle(text)

    def image_pixels(self, half_extents):
        if half_extents != self.half_extents:
            half_x, half_y = half_extents
            self.plot.setRange(xRange=(-half_x, half_x), yRange=(-half_y, half_y), padding=0)
            self.half_extents = half_extents
        # Exclude axes, legend and letterboxing. Equal world-space aspect means
        # the image need not occupy the entire rectangular ViewBox.
        rect = self.plot.getViewBox().sceneBoundingRect()
        half_x, half_y = half_extents
        scale = min(rect.width() / (2 * half_x), rect.height() / (2 * half_y))
        ratio = self.devicePixelRatioF()
        return (max(1, round(2 * half_x * scale * ratio)),
                max(1, round(2 * half_y * scale * ratio)))

    def show_density(self, density, half_extents):
        self.density = density
        # Zero density has no finite logarithm. Keep it below every selectable
        # level so empty cells always match the background, even at low limits.
        log_density = np.full_like(density, -100)  # Controls allow -80 through 80.
        np.log10(density, out=log_density, where=density > 0)
        self.image.setImage(log_density, autoLevels=False)
        half_x, half_y = half_extents
        self.image.setRect(QtCore.QRectF(-half_x, -half_y, 2 * half_x, 2 * half_y))
        self.image.setVisible(True)

    def set_palette(self, name, floor):
        if name == "gray":
            palette = pg.ColorMap([0, 1], [(0, 0, 0), (255, 255, 255)])
        else:
            palette = pg.colormap.get(name)
        colors = palette.map(np.linspace(floor / 100, 1, 256), mode="byte")
        self.legend.setColorMap(pg.ColorMap(np.linspace(0, 1, 256), colors))
        color = tuple(int(x) for x in colors[0, :3])
        self.setBackground(color)
        self.plot.getViewBox().setBackgroundColor(color)


class PlotFullscreen(QtWidgets.QWidget):
    """Temporarily host the existing plots and controls, without the application UI."""

    def __init__(self, viewer, mode):
        super().__init__(viewer.window(), QtCore.Qt.WindowType.Window)
        self.viewer, self.mode = viewer, mode
        self.dragging = False
        self.zooming = False
        self.press_position = None
        self.controls_visible = False
        self.restored = False
        self.setAttribute(QtCore.Qt.WidgetAttribute.WA_DeleteOnClose)
        self.setWindowTitle("Histogram" if mode == "density" else "Particles")
        self.setStyleSheet(viewer.styleSheet() + "QFrame#plotControls { border: 1px solid #34465a; border-radius: 6px; }")
        self.original_sizes = viewer.splitter.sizes()
        self.stack = QtWidgets.QStackedWidget()
        layout = QtWidgets.QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.addWidget(self.stack)
        self.stack.addWidget(viewer.view)
        self.stack.addWidget(viewer.density)
        self.stack.setCurrentWidget(viewer.density if mode == "density" else viewer.view)

        self.controls = QtWidgets.QFrame(self)
        self.controls.setObjectName("plotControls")
        controls = QtWidgets.QVBoxLayout(self.controls)
        panels = [viewer.warning_bar, viewer.playback_panel, viewer.camera_panel, viewer.density_controls]
        self.control_positions = [(viewer.layout().indexOf(panel), panel) for panel in panels]
        for panel in panels:
            visible = not panel.isHidden()
            viewer.layout().removeWidget(panel)
            controls.addWidget(panel)
            panel.setVisible(visible)
        viewer.density_controls.setVisible(mode == "density")
        row = QtWidgets.QHBoxLayout()
        self.histogram_toggle = QtWidgets.QCheckBox("Histogram")
        self.histogram_toggle.setChecked(mode == "density")
        self.histogram_toggle.toggled.connect(self.set_density_mode)
        row.addWidget(self.histogram_toggle)
        hint = "Drag: pan · Shift+drag: orbit" if mode == "density" else "Drag: orbit · Ctrl+drag: pan"
        self.gesture_hint = QtWidgets.QLabel(hint + " · Click image: controls · Esc: exit")
        row.addWidget(self.gesture_hint, 1)
        viewer.add_button(row, "Exit fullscreen", self.close)
        controls.addLayout(row)
        self.zoom_timer = QtCore.QTimer(self)
        self.zoom_timer.setSingleShot(True)
        self.zoom_timer.setInterval(200)
        self.zoom_timer.timeout.connect(self.finish_zoom)
        for plot in (viewer.view, viewer.density.viewport()):
            plot.installEventFilter(self)
        self.shortcuts = []
        for key, action in (("Escape", self.close), ("F", self.close), ("Space", viewer.toggle_playback),
                            ("R", lambda: viewer.select_view("Home"))):
            shortcut = QtGui.QShortcut(QtGui.QKeySequence(key), self)
            shortcut.activated.connect(action)
            self.shortcuts.append(shortcut)
        self.show_controls(False)

    def show_controls(self, visible):
        self.controls_visible = visible
        self.controls.setVisible(visible)
        self.viewer.view.orientation.setVisible(visible)
        self.viewer.view.scale_label.setVisible(visible)
        self.viewer.density.set_plot_only(self.mode == "density" and not visible)
        self.viewer.particle_fullscreen_button.hide()
        self.viewer.density_fullscreen_button.hide()
        self.position_controls()

    def set_density_mode(self, enabled):
        self.mode = "density" if enabled else "particles"
        self.zoom_timer.stop()
        self.zooming = False
        for toggle in (self.histogram_toggle, self.viewer.density_toggle):
            blocker = QtCore.QSignalBlocker(toggle)
            toggle.setChecked(enabled)
            del blocker
        self.setWindowTitle("Histogram" if enabled else "Particles")
        self.viewer.density_controls.setVisible(enabled)
        self.stack.setCurrentWidget(self.viewer.density if enabled else self.viewer.view)
        hint = "Drag: pan · Shift+drag: orbit" if enabled else "Drag: orbit · Ctrl+drag: pan"
        self.gesture_hint.setText(hint + " · Click image: controls · Esc: exit")
        self.show_controls(self.controls_visible)
        self.viewer.invalidate_density()

    def finish_zoom(self):
        self.zooming = False
        if not self.dragging:
            self.stack.setCurrentWidget(self.viewer.density)
            self.viewer.invalidate_density()

    def position_controls(self):
        if not hasattr(self, "controls"): return
        self.controls.setGeometry(12, max(12, self.height() - self.controls.sizeHint().height() - 12),
                                  max(1, self.width() - 24), self.controls.sizeHint().height())
        self.controls.raise_()

    def resizeEvent(self, event):
        super().resizeEvent(event)
        # Hidden stack pages must use the same camera viewport as the visible
        # histogram, otherwise switching representations changes its projection.
        self.viewer.view.resize(self.width(), self.height())
        self.viewer.density.resize(self.width(), self.height())
        self.position_controls()

    def mouse_event(self, event, event_type=None):
        position = QtCore.QPointF(self.viewer.view.mapFromGlobal(event.globalPosition().toPoint()))
        return QtGui.QMouseEvent(event_type or event.type(), position, event.globalPosition(),
                                event.button(), event.buttons(), event.modifiers())

    def eventFilter(self, watched, event):
        kind = event.type()
        if kind == QtCore.QEvent.Type.Wheel and watched in (self.viewer.view, self.viewer.density.viewport()):
            if self.mode == "density":
                self.zooming = True
                self.stack.setCurrentWidget(self.viewer.view)
                self.zoom_timer.start()
            self.viewer.view.wheelEvent(event)
            return True
        if kind == QtCore.QEvent.Type.MouseButtonPress and event.button() in (QtCore.Qt.MouseButton.LeftButton, QtCore.Qt.MouseButton.MiddleButton):
            self.press_position = event.globalPosition()
            self.last_position = event.globalPosition()
            self.dragging = False
            self.viewer.view.mousePressEvent(self.mouse_event(event))
            return True
        if kind == QtCore.QEvent.Type.MouseMove and self.press_position is not None and event.buttons():
            if not self.dragging:
                if (event.globalPosition() - self.press_position).manhattanLength() < QtWidgets.QApplication.startDragDistance():
                    return True
                self.dragging = True
                if self.mode == "density":
                    # Keep receiving the gesture when hiding the pressed widget.
                    # An application filter also works on Wayland, where explicit
                    # mouse grabs by ordinary windows are unsupported.
                    QtWidgets.QApplication.instance().installEventFilter(self)
                    self.stack.setCurrentWidget(self.viewer.view)
            delta = event.globalPosition() - self.last_position
            self.last_position = event.globalPosition()
            if self.mode == "density":
                if event.modifiers() & QtCore.Qt.KeyboardModifier.ShiftModifier:
                    self.viewer.view.orbit(-delta.x(), delta.y())
                else:
                    self.viewer.view.pan(delta.x(), delta.y(), 0, relative="view")
            else:
                self.viewer.view.mouseMoveEvent(self.mouse_event(event))
            return True
        if kind == QtCore.QEvent.Type.MouseButtonRelease and self.press_position is not None:
            was_dragging = self.dragging
            self.press_position = None
            self.dragging = False
            self.viewer.view.mouseReleaseEvent(self.mouse_event(event))
            QtWidgets.QApplication.instance().removeEventFilter(self)
            if self.mode == "density" and was_dragging and not self.zooming:
                self.stack.setCurrentWidget(self.viewer.density)
                self.viewer.invalidate_density()
            elif not was_dragging and event.button() == QtCore.Qt.MouseButton.LeftButton:
                self.show_controls(not self.controls_visible)
            return True
        return super().eventFilter(watched, event)

    def closeEvent(self, event):
        if not self.restored:
            self.restored = True
            self.zoom_timer.stop()
            viewer = self.viewer
            QtWidgets.QApplication.instance().removeEventFilter(self)
            for plot in (viewer.view, viewer.density.viewport()): plot.removeEventFilter(self)
            warning_visible = not viewer.warning_bar.isHidden()
            for plot in (viewer.view, viewer.density):
                self.stack.removeWidget(plot)
                viewer.splitter.addWidget(plot)
            for index, panel in sorted(self.control_positions, key=lambda item: item[0]):
                self.controls.layout().removeWidget(panel)
                panel.setParent(viewer)
                viewer.layout().insertWidget(index, panel)
            viewer.warning_bar.setVisible(warning_visible)
            viewer.playback_panel.show()
            viewer.camera_panel.show()
            viewer.density_controls.setVisible(viewer.density_toggle.isChecked())
            viewer.view.show()
            viewer.density.setVisible(viewer.density_toggle.isChecked())
            viewer.density.set_plot_only(False)
            viewer.view.orientation.show()
            viewer.view.scale_label.show()
            viewer.splitter.setSizes(self.original_sizes)
            viewer.fullscreen_window = None
            viewer.position_fullscreen_buttons()
            viewer.invalidate_density()
        event.accept()


class ViewerWidget(QtWidgets.QWidget):
    def __init__(self, frames, fps=30, name="Snapshots", metadata=None, bounds=None):
        super().__init__()
        self.frames = frames
        self.count, self.center, self.half_width = bounds if bounds is not None else frame_bounds(frames)
        self.metadata = metadata or {}
        self.snapshot_times = {snapshot["file"]: snapshot["time"]
                               for snapshot in self.metadata.get("snapshots", [])
                               if "file" in snapshot and "time" in snapshot}
        self.playback = Playback(len(frames), fps)
        self.index = 0
        self.positions = read_frame(frames, 0)
        validate_frame(self.positions, self.count)
        self.failed_frame = None
        self.warning_kind = None
        self.fullscreen_window = None
        self.active_plot = "particles"
        self.executor = ThreadPoolExecutor(max_workers=1, thread_name_prefix="density")
        self.density_future = None
        self.density_key = None
        self.density_revision = 0
        self.density_settings = None
        self.density_settings_changed = time.perf_counter()
        self.density_initialized = False
        self.fit_density_pending = True
        self.setWindowTitle(f"Particle viewer — {name}")
        self.resize(1280, 850)
        self.setStyleSheet("QWidget { background: #0b111a; color: #dce7f3; }"
                           "QPushButton, QComboBox, QDoubleSpinBox {"
                           " background: #152131; padding: 3px; border: 1px solid #34465a; }"
                           "QSlider::groove:horizontal { height: 3px; background: #34465a; }"
                           "QSlider::handle:horizontal { background: #65d9ed; width: 9px; margin: -4px 0; }")
        layout = QtWidgets.QVBoxLayout(self)
        header = QtWidgets.QHBoxLayout()
        header.addWidget(QtWidgets.QLabel(f"{name}  ·  {self.count:,} particles"))
        header.addStretch()
        self.density_toggle = QtWidgets.QCheckBox("Projected density")
        self.density_toggle.toggled.connect(self.toggle_density)
        header.addWidget(self.density_toggle)
        layout.addLayout(header)
        self.warning_bar = QtWidgets.QWidget()
        warning_layout = QtWidgets.QHBoxLayout(self.warning_bar)
        self.warning_label = QtWidgets.QLabel()
        self.warning_label.setTextFormat(QtCore.Qt.TextFormat.PlainText)
        self.warning_label.setWordWrap(True)
        self.warning_label.setStyleSheet("color: #f3be68;")
        warning_layout.addWidget(self.warning_label, 1)
        self.retry_button = self.add_button(warning_layout, "Retry frame", self.retry_frame)
        self.add_button(warning_layout, "Dismiss", self.warning_bar.hide)
        layout.addWidget(self.warning_bar)
        self.warning_bar.hide()

        self.view = ParticleView(self.center, self.half_width)
        self.points = gl.GLScatterPlotItem(pos=self.positions, size=2.0, pxMode=True,
                                          color=(1, 1, 1, 0.35), glOptions="translucent")
        # Standard alpha blending, no depth rejection: all points contribute.
        # This is a visual aid, not a quantitative density representation.
        self.points.setGLOptions({GL.GL_DEPTH_TEST: False, GL.GL_BLEND: True,
                                 GL.GL_CULL_FACE: False,
                                 "glBlendFuncSeparate": (GL.GL_SRC_ALPHA, GL.GL_ONE_MINUS_SRC_ALPHA,
                                                         GL.GL_ONE, GL.GL_ONE_MINUS_SRC_ALPHA)})
        self.view.addItem(self.points)
        self.paths = []
        if isinstance(frames, np.ndarray) and self.count <= 16 and len(frames) > 1:
            for particle in range(self.count):
                path = gl.GLLinePlotItem(pos=frames[:, particle, :], color=(1, 1, 1, 0.2),
                                         width=1, glOptions="translucent")
                self.view.addItem(path)
                self.paths.append(path)
        self.density = DensityPanel()
        self.density.hide()
        self.splitter = QtWidgets.QSplitter()
        self.splitter.addWidget(self.view)
        self.splitter.addWidget(self.density)
        self.splitter.setSizes([700, 550])
        layout.addWidget(self.splitter, 1)
        self.particle_fullscreen_button = self.plot_fullscreen_button(self.view, "particles")
        self.density_fullscreen_button = self.plot_fullscreen_button(self.density.viewport(), "density")
        self.view.installEventFilter(self)
        self.density.viewport().installEventFilter(self)

        self.playback_panel = QtWidgets.QWidget()
        playback_controls = QtWidgets.QHBoxLayout(self.playback_panel)
        playback_controls.setContentsMargins(0, 0, 0, 0)
        self.play_button = self.add_button(playback_controls, "Pause", self.toggle_playback)
        self.frame_slider = QtWidgets.QSlider(QtCore.Qt.Orientation.Horizontal)
        self.frame_slider.setRange(0, len(frames) - 1)
        self.frame_slider.valueChanged.connect(self.seek)
        playback_controls.addWidget(self.frame_slider, 1)
        self.frame_label = QtWidgets.QLabel()
        playback_controls.addWidget(self.frame_label)
        playback_controls.addWidget(QtWidgets.QLabel("Speed"))
        self.speed = QtWidgets.QComboBox()
        for value in (0.25, 0.5, 1, 2, 4):
            self.speed.addItem(f"{value:g}×", value)
        self.speed.setCurrentIndex(2)
        self.speed.setToolTip("Playback speed relative to the configured frame rate")
        self.speed.currentIndexChanged.connect(lambda _: self.change_speed(self.speed.currentData()))
        playback_controls.addWidget(self.speed)
        layout.addWidget(self.playback_panel)

        self.camera_panel = QtWidgets.QWidget()
        camera_controls = QtWidgets.QHBoxLayout(self.camera_panel)
        camera_controls.setContentsMargins(0, 0, 0, 0)
        for view_name in VIEWS:
            self.add_button(camera_controls, view_name, lambda checked=False, n=view_name: self.select_view(n))
        self.add_button(camera_controls, "Fit all", self.fit_all)
        self.add_button(camera_controls, "Focus central 98%", self.focus_central)
        camera_controls.addStretch()
        self.size = self.add_spin(camera_controls, "Size px", 0.5, 16, 2, 0.5)
        self.opacity = self.add_spin(camera_controls, "Opacity", 0.01, 1, 0.35, 0.05)
        self.color = QtWidgets.QComboBox()
        for label, color in (("White", "#ffffff"), ("Cyan", "#65d9ed"),
                             ("Gold", "#f3be68"), ("Violet", "#b29aff")):
            self.color.addItem(label, color)
        camera_controls.addWidget(self.color)
        self.size.valueChanged.connect(self.particle_style)
        self.opacity.valueChanged.connect(self.particle_style)
        self.color.currentIndexChanged.connect(self.particle_style)
        layout.addWidget(self.camera_panel)

        self.density_controls = QtWidgets.QWidget()
        density_controls = QtWidgets.QHBoxLayout(self.density_controls)
        density_controls.setContentsMargins(0, 0, 0, 0)
        self.grid = QtWidgets.QComboBox()
        self.grid.addItem("Auto", None)
        for value in (128, 256, 512, 1024, 2048):
            self.grid.addItem(str(value), value)
        density_controls.addWidget(QtWidgets.QLabel("Grid"))
        density_controls.addWidget(self.grid)
        self.smoothing = self.add_spin(density_controls, "Smooth px", 0, 32, 1, 0.25)
        self.smoothing.setToolTip("Approximate Gaussian RMS in displayed device pixels; 0 disables smoothing")
        self.palette = QtWidgets.QComboBox()
        self.palette.addItems(COLORMAPS)
        density_controls.addWidget(self.palette)
        self.background = self.add_spin(density_controls, "BG %", 0, 75, 15, 1)
        self.background.setDecimals(0)
        self.low = self.add_spin(density_controls, "log10 min", -80, 80, -2, 0.25)
        self.high = self.add_spin(density_controls, "max", -80, 80, 2, 0.25)
        self.add_button(density_controls, "Fit 1–99.5%", self.fit_density)
        density_controls.addStretch()
        self.grid.currentIndexChanged.connect(self.invalidate_density)
        self.smoothing.valueChanged.connect(self.invalidate_density)
        self.palette.currentTextChanged.connect(self.change_palette)
        self.background.valueChanged.connect(self.change_palette)
        self.low.valueChanged.connect(self.change_levels)
        self.high.valueChanged.connect(self.change_levels)
        self.density_controls.hide()
        layout.addWidget(self.density_controls)

        layout.addWidget(QtWidgets.QLabel("Drag: orbit · Ctrl+drag / middle: pan · Wheel: zoom · Space: pause · R: home · F: fullscreen"))
        self.renderer_label = QtWidgets.QLabel("OpenGL: waiting for context")
        self.status_bar = QtWidgets.QStatusBar(self)
        self.statusBar().addPermanentWidget(self.renderer_label)
        layout.addWidget(self.status_bar)
        self.view.renderer_ready.connect(self.renderer_label.setText)
        self.view.camera_changed.connect(self.invalidate_density)
        self.shortcuts = []
        for key, callback in (("Space", self.toggle_playback), ("R", lambda: self.select_view("Home")),
                              ("F", self.fullscreen)):
            shortcut = QtGui.QShortcut(QtGui.QKeySequence(key), self)
            shortcut.activated.connect(callback)
            self.shortcuts.append(shortcut)
        self.change_palette()
        self.select_view("Home")
        self.draw_frame(0)
        self.play_button.setEnabled(len(frames) > 1)
        self.frame_slider.setEnabled(len(frames) > 1)
        self.speed.setEnabled(len(frames) > 1)
        if not self.playback.playing:
            self.play_button.setText("Play")
        self.timer = QtCore.QTimer(self)
        self.timer.setTimerType(QtCore.Qt.TimerType.PreciseTimer)
        self.timer.timeout.connect(self.tick)
        self.timer.start(max(1, min(16, round(1000 / fps))))

    @staticmethod
    def add_button(layout, text, callback):
        button = QtWidgets.QPushButton(text)
        button.clicked.connect(callback)
        layout.addWidget(button)
        return button

    @staticmethod
    def add_spin(layout, label, low, high, initial, step):
        layout.addWidget(QtWidgets.QLabel(label))
        spin = QtWidgets.QDoubleSpinBox()
        spin.setRange(low, high)
        spin.setSingleStep(step)
        spin.setValue(initial)
        spin.setKeyboardTracking(False)
        spin.setMaximumWidth(85)
        layout.addWidget(spin)
        return spin

    def particle_style(self, *_):
        color = QtGui.QColor(self.color.currentData())
        color.setAlphaF(self.opacity.value())
        self.points.setData(size=self.size.value(), color=color.getRgbF())
        for path in self.paths:
            path.setData(color=(*color.getRgbF()[:3], 0.2))

    def fit_camera(self, center, half_width):
        # Enclose a bounding sphere for every orientation and viewport aspect.
        aspect = max(1, self.view.height()) / max(1, self.view.width())
        half_x = math.sqrt(3) * half_width / min(1, aspect)
        distance = half_x / math.tan(math.radians(self.view.opts["fov"]) / 2)
        self.view.setCameraPosition(pos=QtGui.QVector3D(*center), distance=distance)

    def fit_all(self):
        self.fit_camera(self.center, self.half_width)

    def focus_central(self):
        low, high = np.percentile(self.positions, (1, 99), axis=0)
        self.fit_camera((low + high) / 2, max(float(np.max(high - low)) / 2, 1e-6) * 1.15)

    def select_view(self, name):
        elevation, azimuth = VIEWS[name]
        self.view.setCameraPosition(elevation=elevation, azimuth=azimuth)
        if name == "Home":
            self.fit_all()

    def plot_fullscreen_button(self, parent, mode):
        button = QtWidgets.QToolButton(parent)
        # Draw the corners explicitly so the icon does not depend on font glyphs.
        pixmap = QtGui.QPixmap(40, 40)
        pixmap.setDevicePixelRatio(2)
        pixmap.fill(QtCore.Qt.GlobalColor.transparent)
        painter = QtGui.QPainter(pixmap)
        painter.setPen(QtGui.QPen(QtGui.QColor("#dce7f3"), 2))
        for x, y, dx, dy in ((3, 3, 1, 1), (17, 3, -1, 1), (3, 17, 1, -1), (17, 17, -1, -1)):
            painter.drawLine(x, y, x + dx * 5, y)
            painter.drawLine(x, y, x, y + dy * 5)
        painter.end()
        button.setIcon(QtGui.QIcon(pixmap))
        button.setIconSize(QtCore.QSize(20, 20))
        button.setFixedSize(30, 28)
        name = "Histogram" if mode == "density" else "Particles"
        button.setToolTip(f"{name} only, fullscreen. Click the image for controls; Esc returns.")
        button.setAccessibleName(f"{name} fullscreen")
        button.setStyleSheet("QToolButton { color: #dce7f3; background: #152131; border: 1px solid #34465a; font-size: 18px; }"
                             "QToolButton:hover { background: #24364c; }")
        button.clicked.connect(lambda checked=False: self.fullscreen(mode))
        return button

    def position_fullscreen_buttons(self):
        for button in (self.particle_fullscreen_button, self.density_fullscreen_button):
            button.setVisible(self.fullscreen_window is None)
            parent = button.parentWidget()
            button.move(max(0, parent.width() - button.width() - 10), 10)
            button.raise_()

    def eventFilter(self, watched, event):
        if event.type() == QtCore.QEvent.Type.Resize:
            self.position_fullscreen_buttons()
        elif event.type() == QtCore.QEvent.Type.MouseButtonPress and self.fullscreen_window is None:
            self.active_plot = "particles" if watched is self.view else "density"
        return super().eventFilter(watched, event)

    def fullscreen(self, mode=None):
        if self.fullscreen_window is not None:
            self.fullscreen_window.close()
            return
        self.active_plot = mode or self.active_plot
        self.fullscreen_window = PlotFullscreen(self, self.active_plot)
        self.fullscreen_window.showFullScreen()
        self.fullscreen_window.raise_()
        self.fullscreen_window.activateWindow()

    def statusBar(self):
        return self.status_bar

    def toggle_playback(self):
        if not self.playback.playing and self.failed_frame is not None:
            if not self.retry_frame(): return
        self.playback.playing = not self.playback.playing and len(self.frames) > 1
        self.playback.last_tick = time.perf_counter()
        self.play_button.setText("Pause" if self.playback.playing else "Play")

    def change_speed(self, value):
        self.playback.speed = value
        self.playback.last_tick = time.perf_counter()

    def seek(self, index):
        self.playback.seek(index)
        self.draw_frame(index)

    def draw_frame(self, index):
        try:
            positions = read_frame(self.frames, index)
            validate_frame(positions, self.count)
        except (OSError, ValueError, EOFError) as error:
            self.frame_error(index, error)
            return False
        # Commit only after successful loading/validation. The last valid frame
        # remains visible and selected if a snapshot disappears or changes.
        self.index = index
        self.positions = positions
        self.points.setData(pos=self.positions)
        blocker = QtCore.QSignalBlocker(self.frame_slider)
        self.frame_slider.setValue(index)
        del blocker
        self.failed_frame = None
        if self.warning_kind == "frame":
            self.warning_bar.hide()
            self.warning_kind = None
        text = f"Frame {index + 1:,} / {len(self.frames):,}"
        frame = self.frames[index]
        if isinstance(frame, Path) and frame.name in self.snapshot_times:
            text += f" · t={self.snapshot_times[frame.name]:.5g} T0"
        self.frame_label.setText(text)
        return True

    def show_warning(self, text, kind):
        self.warning_kind = kind
        self.warning_label.setText(text)
        self.retry_button.setVisible(kind == "frame")
        self.warning_bar.show()
        if self.fullscreen_window is not None: self.fullscreen_window.show_controls(True)

    def frame_error(self, index, error):
        self.playback.playing = False
        self.playback.seek(self.index)
        self.play_button.setText("Play")
        blocker = QtCore.QSignalBlocker(self.frame_slider)
        self.frame_slider.setValue(self.index)
        del blocker
        self.failed_frame = index
        source = self.frames[index]
        description = str(source) if isinstance(source, Path) else f"Frame {index + 1}"
        reason = "Snapshot is missing" if isinstance(error, FileNotFoundError) else f"Snapshot cannot be read: {error}"
        self.show_warning(f"Playback paused — {reason}.\n{description}\n"
                          "Restore the snapshot and retry, choose a valid frame, or open another run.", "frame")

    def retry_frame(self):
        if self.failed_frame is None: return True
        index = self.failed_frame
        if not self.draw_frame(index): return False
        self.playback.seek(index)
        return True

    def toggle_density(self, enabled):
        if self.fullscreen_window is not None:
            self.fullscreen_window.set_density_mode(enabled)
            return
        self.density.setVisible(enabled)
        self.density_controls.setVisible(enabled)
        self.invalidate_density()

    def invalidate_density(self, *_):
        self.density_revision += 1

    def change_palette(self, *_):
        self.density.set_palette(self.palette.currentText(), self.background.value())

    def change_levels(self, *_):
        low, high = self.low.value(), self.high.value()
        if low < high:
            self.density.legend.setLevels((low, high))
            self.fit_density_pending = False
            self.density_initialized = True
        else:
            self.statusBar().showMessage("Density minimum must be below maximum", 3000)

    def fit_density(self):
        if self.density.density is None:
            self.fit_density_pending = True
            return
        positive = self.density.density[self.density.density > 0]
        if not len(positive):
            self.statusBar().showMessage("No particles in the density viewport; keeping the range", 3000)
            return
        low, high = np.log10(np.percentile(positive, (1, 99.5)))
        if high - low < 0.25:
            low, high = low - 0.5, high + 0.5
        blockers = (QtCore.QSignalBlocker(self.low), QtCore.QSignalBlocker(self.high))
        self.low.setValue(float(low))
        self.high.setValue(float(high))
        del blockers
        self.change_levels()
        self.density_initialized = True

    def update_density(self):
        enabled = (self.density_toggle.isChecked() if self.fullscreen_window is None else
                   self.fullscreen_window.mode == "density" and not self.fullscreen_window.dragging
                   and not self.fullscreen_window.zooming)
        if enabled:
            rows, extents, _ = self.view.density_camera()
            pixels = self.density.image_pixels(extents)
            settings = (self.density_revision, rows.tobytes(), extents, pixels,
                        self.grid.currentData(), self.smoothing.value())
            if settings != self.density_settings:
                self.density_settings = settings
                self.density_settings_changed = time.perf_counter()
                # Do not stretch an old camera's histogram while rebuilding.
                self.density.image.setVisible(False)
                self.density.sampling_label.setText("Updating density…")
        # Exactly one in-flight calculation, no queued frames. If playback
        # outruns density, show the latest completed frame and label its index.
        if self.density_future is not None and self.density_future.done():
            key, job_extents, future = self.density_job
            self.density_future = None
            try:
                density = future.result()
            except Exception as error:
                if self.failed_frame is None:
                    self.show_warning(f"Density view disabled: {error}. Adjust its settings and enable it to retry.", "density")
                else:
                    self.statusBar().showMessage(f"Density view disabled: {error}")
                self.density_toggle.setChecked(False)
                return
            if enabled and key[1] == self.density_settings:
                if self.warning_kind == "density":
                    self.warning_bar.hide()
                    self.warning_kind = None
                self.density.show_density(density, job_extents)
                self.density.set_frame_title(f"Projected number density · frame {key[0] + 1}")
                pixels = key[1][3]
                sigma = key[1][5]
                physical_sigmas = []
                for half_width, cells, pixel_count in zip(job_extents, density.shape[::-1], pixels):
                    widths = smoothing_boxes(sigma * cells / pixel_count)
                    rms_bins = math.sqrt(sum((width * width - 1) / 12 for width in widths))
                    physical_sigmas.append(rms_bins * 2 * half_width / cells)
                scale = f"{physical_sigmas[0]:.3g}"
                if not math.isclose(*physical_sigmas, rel_tol=0.01, abs_tol=1e-15):
                    scale += f" × {physical_sigmas[1]:.3g}"
                self.density.sampling_label.setText(
                    f"Grid {density.shape[1]} × {density.shape[0]} · smoothing σ ≈ {scale} L0")
                if self.fit_density_pending or not self.density_initialized:
                    self.fit_density_pending = False
                    self.fit_density()
        if not enabled or self.density_future is not None:
            return
        # Briefly combine resize/orbit events before starting a new job.
        if time.perf_counter() - self.density_settings_changed < 0.04:
            return
        key = (self.index, self.density_settings)
        if key == self.density_key:
            return
        self.density_key = key
        self.density_future = self.executor.submit(projected_density, self.positions, rows, extents,
                                                   pixels, self.grid.currentData(), self.smoothing.value())
        self.density_job = (key, extents, self.density_future)

    def tick(self):
        index = self.playback.advance()
        if index != self.index:
            self.draw_frame(index)
        self.update_density()

    def shutdown(self):
        if self.fullscreen_window is not None: self.fullscreen_window.close()
        self.timer.stop()
        self.executor.shutdown(wait=False, cancel_futures=True)

    def changeEvent(self, event):
        super().changeEvent(event)
        if (event.type() == QtCore.QEvent.Type.WindowStateChange and self.isWindow()
                and self.isMaximized() and not self.isFullScreen()):
            QtCore.QTimer.singleShot(0, lambda: self.showFullScreen() if self.isMaximized() else None)

    def closeEvent(self, event):
        self.shutdown()
        super().closeEvent(event)


def create_viewer(frames, fps=30, name="Snapshots", metadata=None, bounds=None):
    return ViewerWidget(frames, fps, name, metadata, bounds)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("source", nargs="?", type=Path,
                        default=Path(__file__).resolve().parents[1] / "output" / "circle_demo.npy")
    parser.add_argument("--fps", type=float, default=30, help="Playback snapshot frames per second")
    args = parser.parse_args()
    if not np.isfinite(args.fps) or args.fps <= 0:
        parser.error("--fps must be positive and finite")
    if not args.source.exists():
        parser.error(f"Not found: {args.source}. Run visualization/generate_demo.py first.")
    metadata = None
    try:
        frames = load_frames(args.source)
        metadata_path = args.source / "metadata.json"
        if args.source.is_dir() and metadata_path.exists():
            metadata = json.loads(metadata_path.read_text())
        pg.setConfigOptions(background=BACKGROUND, foreground="#b7c9db", antialias=False)
        QtCore.QCoreApplication.setAttribute(QtCore.Qt.ApplicationAttribute.AA_ShareOpenGLContexts)
        app = QtWidgets.QApplication.instance() or QtWidgets.QApplication([sys.argv[0]])
        viewer = create_viewer(frames, args.fps, args.source.name, metadata)
        viewer.window_escape = QtGui.QShortcut(QtGui.QKeySequence("Escape"), viewer)
        viewer.window_escape.activated.connect(lambda: viewer.showNormal() if viewer.isFullScreen() else None)
    except (ValueError, OSError, EOFError) as error:
        parser.error(str(error))
    viewer.show()
    sys.exit(app.exec())


if __name__ == "__main__":
    main()
