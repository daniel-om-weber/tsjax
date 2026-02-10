"""Generate synthetic test data for tsjax examples.

Run from project root:  python scripts/generate_test_data.py

Regenerates:
  test_data/DampedSinusoids/   (example 03 — classification)
  test_data/MassSpringDamper/  (example 04 — tabular regression)
"""

import shutil
from pathlib import Path

import h5py
import numpy as np

ROOT = Path(__file__).resolve().parent.parent / "test_data"


def generate_damped_sinusoids(out_dir: Path) -> None:
    """Three damping regimes → class label (0, 1, 2).

    Each file: "u" dataset (1000-pt damped sinusoid) + "class" root attribute.
    """
    if out_dir.exists():
        shutil.rmtree(out_dir)

    rng = np.random.default_rng(42)
    damping = {0: 0.02, 1: 0.10, 2: 0.30}

    for split, n_files in [("train", 30), ("valid", 10), ("test", 10)]:
        split_dir = out_dir / split
        split_dir.mkdir(parents=True)
        for i in range(n_files):
            cls = int(rng.integers(0, 3))
            t = np.linspace(0, 10, 1000, dtype=np.float32)
            u = np.exp(-damping[cls] * t) * np.sin(
                2 * np.pi * t + rng.uniform(0, 2 * np.pi)
            )
            with h5py.File(split_dir / f"{i:03d}.h5", "w") as f:
                f.create_dataset("u", data=u.astype(np.float32))
                f.attrs["class"] = cls

    print(f"DampedSinusoids: {out_dir}  (30 train / 10 valid / 10 test)")


def generate_mass_spring_damper(out_dir: Path) -> None:
    """Mass-spring-damper features → stiffness.

    Each file: root attributes only (peak_freq, gain_db, phase_margin, stiffness).
    """
    if out_dir.exists():
        shutil.rmtree(out_dir)

    rng = np.random.default_rng(42)

    for split, n_files in [("train", 60), ("valid", 20), ("test", 20)]:
        split_dir = out_dir / split
        split_dir.mkdir(parents=True)
        for i in range(n_files):
            stiffness = rng.uniform(10.0, 100.0)
            peak_freq = np.sqrt(stiffness) / (2 * np.pi) + rng.normal(0, 0.1)
            gain_db = 20 * np.log10(1 / stiffness) + rng.normal(0, 0.5)
            phase_margin = 90 - stiffness * 0.3 + rng.normal(0, 2)
            with h5py.File(split_dir / f"{i:03d}.h5", "w") as f:
                f.attrs["peak_freq"] = peak_freq
                f.attrs["gain_db"] = gain_db
                f.attrs["phase_margin"] = phase_margin
                f.attrs["stiffness"] = stiffness

    print(f"MassSpringDamper: {out_dir}  (60 train / 20 valid / 20 test)")


if __name__ == "__main__":
    generate_damped_sinusoids(ROOT / "DampedSinusoids")
    generate_mass_spring_damper(ROOT / "MassSpringDamper")
