"""Generate three circular paths for checking the snapshot viewer."""

import argparse
from pathlib import Path

import numpy as np


def generate_demo(frame_count=240):
    angle = np.linspace(0, 2 * np.pi, frame_count, endpoint=False)
    positions = np.zeros((frame_count, 3, 3), dtype=np.float32)
    # One particle per plane, with different phases so they start apart.
    positions[:, 0, 0] = np.cos(angle)
    positions[:, 0, 1] = np.sin(angle)
    positions[:, 1, 0] = np.cos(angle + 2 * np.pi / 3)
    positions[:, 1, 2] = np.sin(angle + 2 * np.pi / 3)
    positions[:, 2, 1] = np.cos(angle + 4 * np.pi / 3)
    positions[:, 2, 2] = np.sin(angle + 4 * np.pi / 3)
    return positions


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("output", nargs="?", type=Path,
                        default=Path(__file__).resolve().parents[1] / "output" / "circle_demo.npy")
    args = parser.parse_args()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    # Exclusive creation preserves any existing data at this path.
    with args.output.open("xb") as destination:
        np.save(destination, generate_demo(), allow_pickle=False)
    print(f"Wrote {args.output}: 240 frames, 3 particles, float32")
    print("Particle 0: XY, particle 1: XZ, particle 2: YZ")


if __name__ == "__main__":
    main()
