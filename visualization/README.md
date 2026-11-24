# Simulation studio and particle viewer

With your project venv active, launch the complete application from the project root:

```bash
python -m pip install -r visualization/requirements.txt
g++ -std=c++20 -O3 -march=native -fopenmp -Wall -Wextra \
    src/main.cpp src/octree.cpp src/physics.cpp -o simulation_release
python visualization/app.py
```

Use **Setup & Preview** to add, duplicate or remove objects. Available distributions
are a uniform 3D cube, a Gaussian 2D disk, a Gaussian 3D cloud, and the rotating
stellar disk plus bulge. Gaussian models sample independent Gaussian coordinates;
they are not the earlier half-normal-radius model. Each object has its own count,
total mass, seed, position, bulk velocity and XYZ rotation (degrees, applied in that
order). Lengths are in L0 = 10 kpc, masses in M0 = 10¹¹ solar masses, velocities in
L0/T0. Uniform scale is half-width; Gaussian scale is coordinate sigma; galaxy
scale is exponential disk length. Galaxy thickness uses sech²(z/height).
The sidebar separates **Structure**, **Motion**, and **Run settings**. Hover over
a setting or its label for units, interpretation and model assumptions. Galaxy-only
controls are hidden for cloud distributions; the scale label follows the model.

Generate a preview before Run becomes available. Generation uses the C++ simulator;
changing initial conditions, gravity settings, threads or executable invalidates
the preview. Timing and output sampling can be changed without regenerating it.
Run uses the exact
prepared float64 positions, velocities and masses. Sampled velocity vectors show
velocity multiplied by 0.05 T0. Galaxy velocities use the existing approximate
Jeans model. Cloud models offer rest, virialized random motion or approximate
rotation; these are toy initial conditions, not equilibrium galaxy models. Each
object is centered and internally virialized before adding its position and bulk
velocity (rest objects are not virialized). Encounters are not globally virialized.

In **Run settings → Saved snapshots**, choose **Number of frames** to save an
exact count including the initial and final snapshots. The default is 201 frames
over 2000 integration steps. Sampling does not change the timestep or duration
(steps × timestep). Requested times are rounded to the nearest integration step;
at most steps+1 frames can be saved. Zero steps produces one frame. Alternatively
choose **Every N integration steps** for the original interval-based output.
Older saved configurations retain that mode. Saved setup JSON adds `frames` only
in count mode; the CLI equivalent is `--frames N`, which overrides the interval.

Set run name/output parent in **Simulation**. Runs use the chosen local executable;
the interface remains responsive and shows steps, physical time, elapsed time,
estimated remaining time and logs. Cancellation terminates the child process;
partial output is retained, marked cancelled and not offered as completed playback.
Closing the application cancels an active process. Existing run directories are
never overwritten. Choose a different name to rerun a prepared state.

GUI-created runs contain `configuration.json`, `scene.txt`, `initial_state.bin`,
`initial_metadata.json`, `status.json`, `simulation.log`, and a `snapshots/` directory
with the unchanged `.npy` snapshots and simulation `metadata.json`. The private
preparation format is `NBODYIC1`, little-endian uint64 count, then seven little-endian
float64 values per particle: x/y/z, vx/vy/vz, mass. It is separate from playback
snapshots. The C++ CLI also supports `--scene FILE`, `--prepare-only`, and
`--initial-state FILE`; existing CLI usage still works.

**Saved Runs** lists new GUI runs and existing flat snapshot directories. Open a
completed run for embedded playback, or use its saved setup to prepare a new run.
Failed/cancelled/interrupted runs retain their files and status. Snapshot loading
and validation run in a background worker. Unequal object masses are supported;
the heatmap remains projected particle number density, not mass density. The
application needs a native Qt display and runs the simulator locally; remote-job
submission is not implemented.

If a snapshot is deleted or becomes unreadable during playback, the viewer pauses,
keeps the last valid frame selected and visible, and shows one persistent warning.
Restore the file and choose **Retry frame**, scrub to another valid frame, or open
another run. **Play** explicitly retries a failed frame before resuming; it never
keeps retrying a missing file in the background. Dismiss hides the warning without
resuming playback. Replacing a frame with the wrong particle count, invalid shape,
or nonfinite coordinates is also reported. Density failures disable that view and
show a warning while particle viewing remains available.

Snapshot lists in metadata are checked when opening a run. Refresh marks runs with
missing snapshots/metadata as **Damaged**, shows the reason, and disables Open.
Corrupt snapshot contents are checked during loading. Simulator/preparation failures
show one dialog pointing to the output/log; intentional cancellation does not.

## Standalone viewer

From the project root, with your Python virtual environment active:

```bash
python -m pip install -r visualization/requirements.txt
python visualization/viewer.py output/run_001 --fps 30
```

The optional source is a directory of `frame_*.npy` snapshots, one `(N, 3)`
snapshot, or a stacked `(frames, N, 3)` array. Without a source, the viewer opens
`output/circle_demo.npy` (create it with `python visualization/generate_demo.py`).
All frames are validated and scanned for global bounds at startup. Arrays are
memory-mapped; the viewer does not preload a directory into RAM. Optional directory
metadata supplies simulation time; playback still advances at the requested
snapshot FPS, rather than using physical time intervals.

Drag to orbit, Ctrl+drag or middle-drag to pan in the camera plane, and use the
wheel to zoom. Home/R restores the full-animation framing and default orientation.
XY/XZ/YZ change orientation without resetting zoom. Fit all retains orientation;
Focus central 98% frames the current frame's per-coordinate 1st–99th percentiles.
This is a framing aid, not a selection or change to the data. Space pauses;
F toggles fullscreen; Escape leaves fullscreen. Point size is a pixel diameter,
not a physical radius. Opacity is visual alpha, not a measured density.
A small corner triad shows world X/Y/Z orientation; the view width/height is
shown in L0. There are no world-space grid lines or enclosing cube.

Projected density opens a separate panel synchronized to the 3D camera. Its
horizontal and vertical coordinates are camera-right and camera-up offsets from
the camera center, in simulation length units L0. Rotation, pan, and zoom are
controlled from the particle panel. This is number per projected L0² integrated
through the entire snapshot, not 3D density or mass density. The legend and range
controls show log10(number / L0²). Empty cells use the lowest color. The range is
fitted once to the initial displayed positive-density 1st–99.5th percentiles, then
held fixed across playback/camera/grid changes. Fit 1–99.5% explicitly refits the
last displayed density frame. Manual min/max override it. BG truncates the palette;
it does not change density. Grid defaults to Auto: approximately one bin per
displayed image device pixel, capped at 4096 on the longer dimension. It follows
the density panel's actual size and display scaling; manual 128–2048 grids remain
available. Camera changes re-bin the visible region, rather than magnifying an
old histogram. Smooth px is an approximate Gaussian RMS in device pixels, using
three box filters per axis. The applied physical scale is shown below the image;
very small widths can round to zero on coarse manual grids. Padded
binning includes particles just outside the viewport during smoothing. Smooth
image interpolation reduces visible cell edges but does not add particle data.

One background job computes density; Qt and OpenGL updates stay on the GUI thread.
If calculation lags playback, the density panel labels the latest completed frame.
There is no unbounded work queue. Limits, palette, point styling, and other
unrelated controls do not rerun density binning. A running job is discarded if
its camera/grid settings are obsolete. Rapid camera/resize changes are combined
for 40 ms; the old image is hidden until a matching result
is ready. A job running on close is allowed to finish; no additional jobs are
submitted. Particle rendering uses one persistent GLScatterPlotItem; the density
uses one persistent row-major float32 ImageItem.

The startup log and status bar identify the actual OpenGL renderer. `llvmpipe` or
`softpipe` means software rendering, even if the window works. In WSL2, use WSLg
and a current Windows GPU driver supporting WSL acceleration. Microsoft's supported
setup is documented at https://learn.microsoft.com/windows/wsl/tutorials/gui-apps.
Run `wsl --update` in Windows if needed; a subsequent `wsl --shutdown` stops all WSL
sessions, so save work first. The viewer does not change graphics/system settings.
Alternatively install the requirements in a native Windows Python venv and open
the same snapshots there. The application needs a working native Qt display and
OpenGL context; Qt's offscreen platform is not a substitute for testing GL rendering.

Implementation references: PyQtGraph's GLViewWidget/GLScatterPlotItem APIs and
https://pyqtgraph.readthedocs.io/en/pyqtgraph-0.14.0/api_reference/graphicsItems/imageitem.html.
The viewer uses the bundled colormaps and does not require Matplotlib or WebAgg.
