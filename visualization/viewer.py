"""Play NumPy particle snapshots in a rotatable Matplotlib 3D view."""

import argparse
import importlib.util
import json
import time
from pathlib import Path

import matplotlib
import numpy as np
from mpl_toolkits.mplot3d import Axes3D
from mpl_toolkits.mplot3d import proj3d
from matplotlib.colors import ListedColormap, LogNorm
from matplotlib.image import AxesImage

BACKGROUND = "#0b111a"
FOREGROUND = "#dce7f3"
MUTED = "#7f93aa"
PANEL = "#152131"
ACCENT = "#65d9ed"
PARTICLE_COLOR = "#ffffff"
PARTICLE_SIZE = 1.0  # Marker area in points squared; independent of simulation units.
HOME_VIEW = (24, -55)
PLANE_VIEWS = {"Home": HOME_VIEW, "XY": (90, -90), "XZ": (0, -90), "YZ": (0, 0)}
HEATMAP_RESOLUTION = 512
DENSITY_COLORMAPS = ("inferno", "magma", "viridis", "plasma", "cividis", "gray")


class ViewportAxes3D(Axes3D):
    """Use the entire rectangular viewport without stretching the 3D scene."""

    def apply_aspect(self, position=None):
        # Axes3D normally shrinks its allocated rectangle to a centered square.
        if position is None:
            position = self.get_position(original=True)
        self._set_position(position, which="active")

    def get_proj(self):
        projection = super().get_proj()
        width, height = self.get_window_extent().size
        if width > 0 and height > 0:
            scale = min(width, height)
            projection[0] *= scale / width
            projection[1] *= scale / height
        return projection


def projected_density(positions, axes, projection, resolution=HEATMAP_RESOLUTION, smoothing=1):
    """Return lightly smoothed particle counts per simulation area on the camera plane."""
    def project(points):
        x, y, _ = proj3d.proj_transform(*points.T, projection)
        pixels = axes.transData.transform(np.column_stack((x, y)))
        return axes.transAxes.inverted().transform(pixels)

    width, height = axes.get_window_extent().size
    # Keep cells approximately square on screen, including in a wide viewport.
    columns = max(1, round(resolution * width / max(width, height)))
    rows = max(1, round(resolution * height / max(width, height)))
    screen = project(positions)
    counts, _, _ = np.histogram2d(screen[:, 1], screen[:, 0], bins=(rows, columns),
                                 range=((0, 1), (0, 1)))

    # For our orthographic camera, this Jacobian is constant across the image.
    # Convert counts per screen cell to counts per world-space projected area,
    # so zooming does not arbitrarily change what a density color means.
    origin = np.array([np.mean(axes.get_xlim()), np.mean(axes.get_ylim()), np.mean(axes.get_zlim())])
    basis = project(np.vstack((origin, origin + np.eye(3))))
    gradients = (basis[1:] - basis[0]).T
    area_scale = np.linalg.norm(np.cross(gradients[0], gradients[1]))

    # Separable [1, 2, 1] filter: lighter Gaussian-like smoothing, with
    # zero outside the visible image, using NumPy rather than an extra library.
    for _ in range(smoothing):
        padded = np.pad(counts, ((0, 0), (1, 1)))
        counts = (padded[:, :-2] + 2 * padded[:, 1:-1] + padded[:, 2:]) / 4
        padded = np.pad(counts, ((1, 1), (0, 0)))
        counts = (padded[:-2] + 2 * padded[1:-1] + padded[2:]) / 4
    return counts * area_scale * rows * columns


class CameraDensityImage(AxesImage):
    """Recompute density only when the frame, camera, or viewport changes."""

    def __init__(self, axes, positions, half_width):
        reference_area = (2 * half_width) ** 2
        super().__init__(axes, origin="lower", extent=(0, 1, 0, 1),
                         transform=axes.transAxes, interpolation="bilinear",
                         cmap=DENSITY_COLORMAPS[0],
                         norm=LogNorm(vmin=1 / reference_area,
                                      vmax=len(positions) * HEATMAP_RESOLUTION ** 2 / reference_area))
        self.positions = positions
        self.frame_version = 0
        self.resolution = HEATMAP_RESOLUTION
        self.smoothing = 1
        self._density_key = None
        self.set_data(np.ma.masked_all((2, 2)))
        self.set_visible(False)

    def set_positions(self, positions):
        self.positions = positions
        self.frame_version += 1

    def update_density(self):
        projection = self.axes.get_proj()
        key = (self.frame_version, self.resolution, self.smoothing,
               projection.tobytes(), self.axes.bbox.bounds,
               self.axes.viewLim.bounds)
        if key != self._density_key:
            density = projected_density(self.positions, self.axes, projection,
                                        self.resolution, self.smoothing)
            # Clamp empty cells to the color floor before interpolation, avoiding
            # the hard boundaries produced by interpolating masked values.
            self.set_data(np.maximum(density, self.norm.vmin))
            # Keep the axes visible in empty regions; their background now
            # matches the color floor, and this alpha is interpolated smoothly.
            self.set_alpha((density > self.norm.vmin).astype(float))
            self._density_key = key

    def draw(self, renderer):
        if not self.get_visible():
            return
        self.update_density()
        super().draw(renderer)


def load_frames(source):
    """Read a stacked demo (F, N, 3), a single frame, or a frame directory."""
    if source.is_dir():
        frames = sorted(source.glob("frame_*.npy"))
        if not frames:
            raise ValueError(f"No frame_*.npy snapshots in {source}")
        return frames
    frames = np.load(source, mmap_mode="r", allow_pickle=False)
    if frames.ndim == 2:
        frames = frames[np.newaxis, ...]
    if frames.ndim != 3 or len(frames) == 0:
        raise ValueError("Expected a snapshot (N, 3) or an animation (frames, N, 3)")
    return frames


def read_frame(frames, index):
    item = frames[index]
    return np.load(item, mmap_mode="r", allow_pickle=False) if isinstance(item, Path) else item


def frame_bounds(frames):
    """Scan once for fixed limits, without retaining all directory snapshots."""
    minimum = np.full(3, np.inf)
    maximum = np.full(3, -np.inf)
    count = len(read_frame(frames, 0))
    for index in range(len(frames)):
        positions = read_frame(frames, index)
        if positions.shape != (count, 3) or count == 0:
            raise ValueError("Every frame must have the same nonempty (N, 3) shape")
        if not np.issubdtype(positions.dtype, np.number) or np.iscomplexobj(positions):
            raise ValueError("Positions must contain real numbers")
        if not np.isfinite(positions).all():
            raise ValueError("Positions must be finite")
        minimum = np.minimum(minimum, positions.min(axis=0))
        maximum = np.maximum(maximum, positions.max(axis=0))
    center = minimum / 2 + maximum / 2
    half_width = max(float(np.max(maximum - minimum)) / 2, 1e-6) * 1.15
    return count, center, half_width


def create_viewer(frames, fps=30, name="Snapshots"):
    import matplotlib.pyplot as plt
    from matplotlib.ticker import FixedLocator, FuncFormatter, MaxNLocator, NullLocator
    from matplotlib.widgets import Button, Slider

    count, center, half_width = frame_bounds(frames)
    figure = plt.figure(figsize=(11, 8), facecolor=BACKGROUND)
    figure.canvas.manager.set_window_title("Particle viewer")
    figure.text(0.025, 0.963, "PARTICLE", color=FOREGROUND, fontsize=12)
    figure.text(0.025, 0.935, name, color=MUTED, fontsize=8)
    figure.text(0.975, 0.963, f"{count:,} particles", ha="right", color=FOREGROUND, fontsize=9)
    figure.text(0.975, 0.935, f"{len(frames):,} frames  /  {fps:g} fps", ha="right",
                color=MUTED, fontsize=8)
    figure.text(0.025, 0.012, "Drag: orbit   |   Scroll: zoom   |   Space: pause   |   R: home   |   F: fullscreen",
                color=MUTED, fontsize=7)

    axes = figure.add_axes([0.015, 0.135, 0.97, 0.775], axes_class=ViewportAxes3D,
                          facecolor=BACKGROUND)
    view_zoom = 1.0
    axes.set_box_aspect((1, 1, 1), zoom=view_zoom)
    axes.set_proj_type("ortho")
    axes.view_init(elev=HOME_VIEW[0], azim=HOME_VIEW[1])
    for axis in (axes.xaxis, axes.yaxis, axes.zaxis):
        axis.set_pane_color((0, 0, 0, 0))
        axis.line.set_color("#34465a")
        axis.set_major_locator(MaxNLocator(nbins=3, prune="both"))
        for artist in (axis.line, axis.label, *axis.get_ticklabels(), *axis.get_ticklines()):
            artist.set_clip_path(axes.patch)
            artist.set_clip_on(True)
    axes.tick_params(colors=MUTED, labelsize=8, pad=1)
    axes.grid(False)
    axes.set(xlim=(center[0] - half_width, center[0] + half_width),
             ylim=(center[1] - half_width, center[1] + half_width),
             zlim=(center[2] - half_width, center[2] + half_width))
    for label, setter in (("X", axes.set_xlabel), ("Y", axes.set_ylabel), ("Z", axes.set_zlabel)):
        setter(label, color=MUTED, labelpad=8)

    first = read_frame(frames, 0)
    points = axes.scatter(first[:, 0], first[:, 1], first[:, 2], c=PARTICLE_COLOR,
                          s=PARTICLE_SIZE, depthshade=False, linewidths=0)
    heatmap = CameraDensityImage(axes, first, half_width)
    axes.add_image(heatmap)
    # Small stacked samples can show full reference paths without extra I/O.
    # Directory playback stays frame-by-frame and does not retain trajectories.
    if isinstance(frames, np.ndarray) and count <= 16 and len(frames) > 1:
        for index in range(count):
            path = frames[:, index, :]
            axes.plot(path[:, 0], path[:, 1], path[:, 2], color=PARTICLE_COLOR,
                      linewidth=1, alpha=0.25)

    status = figure.text(0.71, 0.097, "", ha="right", color=MUTED, fontsize=7)
    button_axes = figure.add_axes([0.025, 0.087, 0.055, 0.027])
    button = Button(button_axes, "Pause", color=BACKGROUND, hovercolor=PANEL)
    button.label.set(color=MUTED, fontsize=8)
    slider_axes = figure.add_axes([0.13, 0.096, 0.43, 0.008], facecolor=BACKGROUND)
    slider = Slider(slider_axes, "Frame", 0, max(len(frames) - 1, 1), valinit=0,
                    valstep=1, valfmt="%d", color=ACCENT, track_color=PANEL,
                    handle_style={"facecolor": FOREGROUND, "edgecolor": BACKGROUND, "size": 5})
    slider.label.set(color=MUTED, fontsize=8)
    slider.valtext.set_visible(False)
    speed_axes = figure.add_axes([0.81, 0.096, 0.13, 0.008], facecolor=BACKGROUND)
    speed = Slider(speed_axes, "Speed", 0.25, 4.0, valinit=1.0, valstep=0.25,
                   valfmt="%.2fx", color=ACCENT, track_color=PANEL,
                   handle_style={"facecolor": FOREGROUND, "edgecolor": BACKGROUND, "size": 5})
    speed.label.set(color=MUTED, fontsize=8)
    speed.valtext.set(color=MUTED, fontsize=8)
    size_axes = figure.add_axes([0.26, 0.054, 0.10, 0.008], facecolor=BACKGROUND)
    size = Slider(size_axes, "Size", 0.25, 16, valinit=PARTICLE_SIZE, valstep=0.25,
                  valfmt="%.2f", color=ACCENT, track_color=PANEL,
                   handle_style={"facecolor": FOREGROUND, "edgecolor": BACKGROUND, "size": 5})
    size.label.set(color=MUTED, fontsize=8)
    size.valtext.set(color=MUTED, fontsize=8)

    def change_size(value):
        points.set_sizes([value])
        figure.canvas.draw_idle()

    def change_color(color):
        points.set_color(color)
        for line in axes.lines:
            line.set_color(color)
        figure.canvas.draw_idle()

    def select_view(name):
        nonlocal view_zoom
        elevation, azimuth = PLANE_VIEWS[name]
        axes.view_init(elev=elevation, azim=azimuth, roll=0)
        if name == "Home":
            view_zoom = 1.0
            axes.set_box_aspect((1, 1, 1), zoom=view_zoom)
            axes.set(xlim=(center[0] - half_width, center[0] + half_width),
                     ylim=(center[1] - half_width, center[1] + half_width),
                     zlim=(center[2] - half_width, center[2] + half_width))
        figure.canvas.draw_idle()

    controls = []
    for index, name in enumerate(PLANE_VIEWS):
        view_axes = figure.add_axes([0.025 + index * 0.045, 0.042, 0.04, 0.027])
        control = Button(view_axes, name, color=BACKGROUND, hovercolor=PANEL)
        control.label.set(color=MUTED, fontsize=8)
        control.on_clicked(lambda _event, view=name: select_view(view))
        controls.append(control)
    color_label = figure.text(0.425, 0.05, "Color", color=MUTED, fontsize=8)
    for index, (name, color) in enumerate((("White", "#ffffff"), ("Cyan", "#65d9ed"),
                                          ("Gold", "#f3be68"), ("Violet", "#b29aff"))):
        color_axes = figure.add_axes([0.47 + index * 0.065, 0.042, 0.06, 0.027])
        control = Button(color_axes, name, color=BACKGROUND, hovercolor=PANEL)
        control.label.set(color=color, fontsize=8)
        control.on_clicked(lambda _event, value=color: change_color(value))
        controls.append(control)

    legend_axes = figure.add_axes([0.33, 0.95, 0.29, 0.008])
    legend = figure.colorbar(heatmap, cax=legend_axes, orientation="horizontal", ticklocation="top")
    density_ticks = [heatmap.norm.vmin, np.sqrt(heatmap.norm.vmin * heatmap.norm.vmax), heatmap.norm.vmax]
    legend.locator = FixedLocator(density_ticks)
    legend.formatter = FuncFormatter(lambda value, _position: f"{value:.2g}")
    legend.minorlocator = NullLocator()
    legend.update_ticks()
    legend.ax.tick_params(colors=MUTED, labelsize=6, length=2, pad=1)
    legend.outline.set_visible(False)
    legend.set_label("Projected particles / unit area (log scale)", color=MUTED, fontsize=6, labelpad=1)
    legend_axes.set_visible(False)
    mode_axes = figure.add_axes([0.755, 0.042, 0.085, 0.027])
    mode = Button(mode_axes, "Heatmap", color=BACKGROUND, hovercolor=PANEL)
    mode.label.set(color=ACCENT, fontsize=8)

    heatmap_sliders = []
    for position, label, low, high, initial, step, format_string in (
        ([0.29, 0.054, 0.08, 0.008], "BG", 0, 75, 15, 1, "%d%%"),
        ([0.47, 0.054, 0.08, 0.008], "Smooth", 0, 32, 1, 1, "%d"),
        ([0.64, 0.054, 0.08, 0.008], "Grid", 128, 2048, HEATMAP_RESOLUTION,
         np.array([128, 256, 512, 1024, 2048]), "%d"),
    ):
        panel = figure.add_axes(position, facecolor=BACKGROUND)
        control = Slider(panel, label, low, high, valinit=initial, valstep=step,
                         valfmt=format_string, color=ACCENT, track_color=PANEL,
                         handle_style={"facecolor": FOREGROUND, "edgecolor": BACKGROUND, "size": 5})
        control.label.set(color=MUTED, fontsize=8)
        control.valtext.set(color=MUTED, fontsize=8)
        panel.set_visible(False)
        control.set_active(False)
        heatmap_sliders.append(control)
    background_level, smoothing_level, grid_size = heatmap_sliders

    def density_state():
        return {"enabled": heatmap.get_visible(), "colormap": heatmap.get_cmap().name,
                "background": matplotlib.colors.to_hex(figure.get_facecolor())}

    def update_background():
        color = heatmap.get_cmap()(0.0) if heatmap.get_visible() else BACKGROUND
        figure.set_facecolor(color)
        for panel in figure.axes:
            panel.set_facecolor(color)
        for control in (button, mode, *controls):
            control.color = color

    def notify_density_state():
        if hasattr(figure.canvas, "send_event"):
            figure.canvas.send_event("density_state", **density_state())

    def density_colormap(name):
        palette = matplotlib.colormaps[name]
        # BG selects the palette's starting color, independent of the density
        # calculation. The legend uses the same mapping and fixed density range.
        start = background_level.val / 100
        return ListedColormap(palette(np.linspace(start, 1, palette.N)), name=name)

    def set_heatmap(enabled):
        if enabled:
            heatmap.set_cmap(density_colormap(heatmap.get_cmap().name))
        heatmap.set_visible(enabled)
        points.set_visible(not enabled)
        for line in axes.lines:
            line.set_visible(not enabled)
        size_axes.set_visible(not enabled)
        size.set_active(not enabled)
        color_label.set_visible(not enabled)
        for control in controls[4:]:
            control.ax.set_visible(not enabled)
            control.set_active(not enabled)
        legend_axes.set_visible(enabled)
        for control in heatmap_sliders:
            control.ax.set_visible(enabled)
            control.set_active(enabled)
        mode.label.set_text("Particles" if enabled else "Heatmap")
        update_background()
        notify_density_state()
        figure.canvas.draw_idle()

    def set_colormap(name):
        if name not in DENSITY_COLORMAPS:
            raise ValueError("Unknown density colormap")
        heatmap.set_cmap(density_colormap(name))
        update_background()
        notify_density_state()
        figure.canvas.draw_idle()

    def change_smoothing(value):
        heatmap.smoothing = int(value)
        figure.canvas.draw_idle()

    def change_grid(value):
        heatmap.resolution = int(value)
        figure.canvas.draw_idle()

    background_level.on_changed(lambda _value: set_colormap(heatmap.get_cmap().name))
    smoothing_level.on_changed(change_smoothing)
    grid_size.on_changed(change_grid)

    mode.on_clicked(lambda _event: set_heatmap(not heatmap.get_visible()))
    size.on_changed(change_size)
    for control in (button, mode, *controls):
        for spine in control.ax.spines.values():
            spine.set_visible(False)
    if len(frames) == 1:
        slider.set_active(False)
        speed.set_active(False)
    playing = True
    frame_position = 0.0
    last_tick = time.perf_counter()
    advancing = False

    def draw_frame(value):
        nonlocal frame_position, last_tick
        index = min(int(value), len(frames) - 1)
        if not advancing:
            frame_position = float(index)
            last_tick = time.perf_counter()
        positions = read_frame(frames, index)
        heatmap.set_positions(positions)
        points.set_offsets(positions[:, :2])
        points.set_3d_properties(positions[:, 2], zdir="z")
        status.set_text(f"FRAME {index + 1:04d} / {len(frames):04d}")
        figure.canvas.draw_idle()

    def toggle_playback(_event=None):
        nonlocal playing, last_tick
        playing = not playing
        last_tick = time.perf_counter()
        button.label.set_text("Pause" if playing else "Play")
        figure.canvas.draw_idle()

    def key_press(event):
        if event.key == " ":
            toggle_playback()
        elif event.key == "r":
            select_view("Home")

    def scroll_zoom(event):
        nonlocal view_zoom
        if event.inaxes != axes or event.step == 0:
            return
        # Scale the projected scene, including the axis lines, rather than
        # shrinking data limits inside a fixed-size axis box.
        view_zoom = float(np.clip(view_zoom * 0.85 ** (-event.step), 0.2, 4.0))
        axes.set_box_aspect((1, 1, 1), zoom=view_zoom)
        figure.canvas.draw_idle()

    def change_speed(_value):
        nonlocal last_tick
        last_tick = time.perf_counter()

    def advance():
        nonlocal frame_position, last_tick, advancing
        now = time.perf_counter()
        elapsed = now - last_tick
        last_tick = now
        if playing and len(frames) > 1:
            # Use elapsed time so slow renders don't dictate playback speed.
            # Frames may be skipped when rendering cannot keep up.
            frame_position = (frame_position + elapsed * fps * speed.val) % len(frames)
            index = int(frame_position)
            if index != int(slider.val):
                advancing = True
                try:
                    slider.set_val(index)
                finally:
                    advancing = False

    slider.on_changed(draw_frame)
    speed.on_changed(change_speed)
    button.on_clicked(toggle_playback)
    figure.canvas.mpl_connect("key_press_event", key_press)
    figure.canvas.mpl_connect("scroll_event", scroll_zoom)
    timer = figure.canvas.new_timer(interval=max(1, round(1000 / fps)))
    timer.add_callback(advance)
    figure.canvas.mpl_connect("close_event", lambda _event: timer.stop())

    def start_timer(_event):
        nonlocal last_tick
        # Start after the first render, when the interactive event loop exists.
        last_tick = time.perf_counter()
        timer.start()
        figure.canvas.mpl_disconnect(start_connection)

    start_connection = figure.canvas.mpl_connect("draw_event", start_timer)
    draw_frame(0)
    # Retain widgets and the timer for the lifetime of the displayed figure.
    density_controls = {"set_enabled": set_heatmap, "set_colormap": set_colormap,
                        "state": density_state, "button": mode,
                        "background": background_level, "smoothing": smoothing_level,
                        "resolution": grid_size}
    return figure, slider, button, timer, speed, size, controls, density_controls


def show_browser(figure, density_controls):
    from matplotlib.backends.backend_webagg import WebAggApplication
    from tornado.web import RequestHandler

    figure.canvas.set_capture_scroll(True)

    class ViewerPage(RequestHandler):
        def initialize(self, url_prefix=""):
            self.prefix = url_prefix

        def get(self, figure_id=None):
            if figure_id is not None and int(figure_id) != figure.number:
                self.set_status(404)
                self.finish()
                return
            self.render(str(Path(__file__).with_name("viewer.html").resolve()),
                        prefix=self.prefix, fig_id=figure.number, colormaps=DENSITY_COLORMAPS)

    class ViewerSocket(WebAggApplication.WebSocket):
        def open(self, fignum):
            super().open(fignum)
            self.send_json({"type": "density_state", **density_controls["state"]()})

        def on_message(self, message):
            event = json.loads(message)
            if event.get("type") == "density_colormap":
                if event.get("value") in DENSITY_COLORMAPS:
                    density_controls["set_colormap"](event["value"])
            else:
                super().on_message(message)

    class ViewerApplication(WebAggApplication):
        SingleFigurePage = ViewerPage
        AllFiguresPage = ViewerPage
        WebSocket = ViewerSocket

    ViewerApplication.initialize(address="127.0.0.1")
    print(f"Open http://127.0.0.1:{ViewerApplication.port} to view particles", flush=True)
    ViewerApplication.start()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("source", nargs="?", type=Path,
                        default=Path(__file__).resolve().parents[1] / "output" / "circle_demo.npy")
    parser.add_argument("--fps", type=float, default=30, help="Playback frames per second")
    args = parser.parse_args()
    if not np.isfinite(args.fps) or args.fps <= 0:
        parser.error("--fps must be positive and finite")
    if not args.source.exists():
        parser.error(f"Not found: {args.source}. Run visualization/generate_demo.py first.")

    # WebAgg works in WSL without a native Tk/Qt installation. An explicitly
    # selected interactive backend is respected, allowing desktop use later.
    if matplotlib.get_backend().lower() == "agg":
        if importlib.util.find_spec("tornado") is None:
            parser.error("Interactive playback needs a backend: install tornado in your venv.")
        matplotlib.use("WebAgg")
    if matplotlib.get_backend().lower() == "webagg":
        matplotlib.rcParams["webagg.address"] = "127.0.0.1"
        matplotlib.rcParams["webagg.open_in_browser"] = False
    try:
        frames = load_frames(args.source)
        viewer = create_viewer(frames, args.fps, args.source.name)
    except (ValueError, OSError) as error:
        parser.error(str(error))
    import matplotlib.pyplot as plt
    if matplotlib.get_backend().lower() == "webagg":
        show_browser(viewer[0], viewer[7])
    else:
        plt.show()
    viewer[3].stop()


if __name__ == "__main__":
    main()
