# %% [markdown]
# # Attitude Estimation from IMU Data
#
# Estimates 3D orientation (quaternions) from raw IMU signals
# (accelerometer + gyroscope) using an RNN model trained with
# quaternion-specific loss functions and NaN-safe computation.
#
# Based on the GeneralAttitudeEstimator pipeline (without TBPTT).
# Configured for the Myon dataset by default.

# %% Imports

from pathlib import Path

import numpy as np
from flax import nnx

from tsjax import (
    RNN,
    Denormalize,
    Learner,
    Normalize,
    NormalizedModel,
    bias_injection,
    create_grain_dls,
    uniform_file_weights,
)
from tsjax.quaternion import (
    abs_inclination,
    nan_safe,
    plot_quaternion_results,
    quaternion_augmentation,
    rms_inclination_deg,
)

if __name__ == "__main__":
    # %% Configuration

    DATA_DIR = Path("~/Development/mdt_strange/Systemidentification/Orientation/Myon").expanduser()

    # Myon split definitions (by filename prefix)
    VALID_PREFIXES = ["14_", "39_", "21_"]
    TEST_PREFIXES = ["29_", "22_", "35_"]

    # IMU signal names (6 input channels, 4 target channels)
    INPUT_SIGNALS = ["acc_x", "acc_y", "acc_z", "gyr_x", "gyr_y", "gyr_z"]
    TARGET_SIGNALS = ["opt_a", "opt_b", "opt_c", "opt_d"]

    # Data
    WIN_SZ = 300  # Window size (samples)
    STP_SZ = 1  # Window step (training)
    BS = 64

    # Model
    HIDDEN_SIZE = 100
    NUM_LAYERS = 2
    CELL_TYPE = nnx.GRUCell

    # Training
    N_EPOCHS = 30
    LR = 3e-3
    N_SKIP = 100  # Skip initial RNN warmup in loss

    # %% [markdown]
    # ## File Splits
    #
    # The Myon dataset has all files in one directory. Train/valid/test splits
    # are defined by filename prefixes (matching the original experiment).

    # %% File splits

    all_files = sorted(DATA_DIR.glob("*.hdf5"))

    def _match(f, prefixes):
        return any(f.name.startswith(p) for p in prefixes)

    valid_files = [f for f in all_files if _match(f, VALID_PREFIXES)]
    test_files = [f for f in all_files if _match(f, TEST_PREFIXES)]
    train_files = [f for f in all_files if f not in set(valid_files + test_files)]

    print(f"Train: {len(train_files)}, Valid: {len(valid_files)}, Test: {len(test_files)}")

    # %% [markdown]
    # ## Augmentations
    #
    # Two training-only augmentations:
    # 1. **Quaternion rotation** — applies a random rotation to both input vectors
    #    (accelerometer ch 0–2, gyroscope ch 3–5) and target quaternions.
    # 2. **Gyroscope bias injection** — adds a small constant offset per sample
    #    to gyroscope channels only (simulates sensor bias drift).

    # %% Augmentations

    quat_aug = quaternion_augmentation(inp_groups=[(0, 2), (3, 5)])
    gyr_bias = bias_injection(std=np.array([0, 0, 0, 0.02, 0.02, 0.02]))

    def augmentation(item, rng):
        item = quat_aug(item, rng)
        return {**item, "u": gyr_bias(item["u"], rng)}

    # %% [markdown]
    # ## Data Pipeline
    #
    # Each HDF5 file contains individual 1D signals (`acc_x`, `gyr_x`, `opt_a`, etc.).
    # `create_grain_dls` stacks them into multi-channel arrays:
    # - `u`: `(win_sz, 6)` — accelerometer + gyroscope
    # - `y`: `(win_sz, 4)` — quaternion `[w, x, y, z]`
    #
    # `uniform_file_weights` ensures each recording contributes equally
    # regardless of its length.

    # %% Data pipeline

    pipeline = create_grain_dls(
        inputs={"u": INPUT_SIGNALS},
        targets={"y": TARGET_SIGNALS},
        train_files=train_files,
        valid_files=valid_files,
        test_files=test_files,
        bs=BS,
        win_sz=WIN_SZ,
        stp_sz=STP_SZ,
        preload=True,
        weights=uniform_file_weights,
        augmentation=augmentation,
        worker_count=0,
    )
    pipeline.n_train_batches = 300

    # %% [markdown]
    # ## Model
    #
    # Multi-layer GRU with input z-score normalization. Output denormalization
    # is identity since quaternions are unit-scale.

    # %% Model
    from tsjax.data.stats import compute_stats

    stats = compute_stats(pipeline.train)
    rnn = RNN(
        input_size=6,
        output_size=4,
        hidden_size=HIDDEN_SIZE,
        num_layers=NUM_LAYERS,
        cell_type=CELL_TYPE,
        rngs=nnx.Rngs(0),
    )
    model = NormalizedModel(
        rnn,
        norm_in=Normalize(6, stats["u"].mean, stats["u"].std),
        norm_out=Denormalize(4),  # Identity — quaternions are unit-scale
    )

    # %% [markdown]
    # ## Training
    #
    # - **Loss**: Mean absolute inclination angle (radians), NaN-safe.
    # - **Metric**: RMS inclination error in degrees, NaN-safe.
    # - **`n_skip`**: Ignores the first 100 predictions to let the RNN build
    #   up hidden state before computing loss.

    # %% Training

    loss_fn = nan_safe(abs_inclination)
    metrics = [nan_safe(rms_inclination_deg)]

    lrn = Learner(
        model,
        pipeline,
        loss_func=loss_fn,
        n_skip=N_SKIP,
        metrics=metrics,
        plot_results_fn=plot_quaternion_results,
    )
    lrn.fit_flat_cos(n_epoch=N_EPOCHS, lr=LR)

    # %% Results

    lrn.show_results(n=2)
