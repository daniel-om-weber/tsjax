"""Validate Grain pipeline against TSFast pipeline numerically.

Usage:
    python -m tsjax.validation --dataset test_data/pinn_var_ic --u u --y x v --win_sz 100
    python -m tsjax.validation --dataset test_data/WienerHammerstein --u u --y y --win_sz 100

NOTE: grain and tsfast/PyTorch cannot coexist in the same process (abseil vs libc++ mutex
conflict). Level 1 uses subprocess to compute TSFast reference values.
"""
from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path

import h5py
import numpy as np


def _find_hdf_files(dataset: Path, split: str) -> list[str]:
    """Find sorted HDF5 files in a split directory."""
    extensions = {'.hdf5', '.h5'}
    return sorted(
        str(p) for p in (dataset / split).rglob('*')
        if p.suffix in extensions
    )


def _get_tsfast_norm_stats(train_files: list[str], signals: list[str]) -> tuple[np.ndarray, np.ndarray]:
    """Compute TSFast norm stats in a subprocess (avoids grain/torch mutex conflict)."""
    script = f"""
import json, numpy as np
from tsfast.datasets.core import extract_mean_std_from_hdffiles
files = {train_files!r}
signals = {signals!r}
mean, std = extract_mean_std_from_hdffiles(files, signals)
print(json.dumps({{'mean': mean.tolist(), 'std': std.tolist()}}))
"""
    result = subprocess.run(
        [sys.executable, '-c', script],
        capture_output=True, text=True, timeout=60,
    )
    if result.returncode != 0:
        raise RuntimeError(f'TSFast subprocess failed:\n{result.stderr}')
    data = json.loads(result.stdout.strip())
    return np.array(data['mean'], dtype=np.float32), np.array(data['std'], dtype=np.float32)


def run_validation(dataset: str, u: list[str], y: list[str], win_sz: int, stp_sz: int = 1):
    """Run 5-level validation comparing Grain vs TSFast."""
    dataset = Path(dataset)
    all_signals = u + y
    valid_stp_sz = win_sz
    all_passed = True

    # ── Level 1: Normalization stats ──
    print('=' * 60)
    print('Level 1: Normalization stats')
    print('=' * 60)

    from .stats import compute_norm_stats
    train_files = _find_hdf_files(dataset, 'train')
    grain_mean, grain_std = compute_norm_stats(train_files, u)

    # TSFast reference (subprocess to avoid mutex conflict)
    tsfast_mean, tsfast_std = _get_tsfast_norm_stats(train_files, u)

    mean_match = np.allclose(grain_mean, tsfast_mean, atol=0)
    std_match = np.allclose(grain_std, tsfast_std, atol=0)
    print(f'  Grain mean:  {grain_mean}')
    print(f'  TSFast mean: {tsfast_mean}')
    print(f'  Mean exact match: {mean_match}')
    print(f'  Grain std:   {grain_std}')
    print(f'  TSFast std:  {tsfast_std}')
    print(f'  Std exact match:  {std_match}')
    l1_pass = mean_match and std_match
    print(f'  LEVEL 1: {"PASS" if l1_pass else "FAIL"}')
    all_passed &= l1_pass

    # ── Level 2: Window count ──
    print()
    print('=' * 60)
    print('Level 2: Window count')
    print('=' * 60)

    from .hdf5_index import HDF5MmapIndex
    from .sources import WindowedHDF5Source

    l2_pass = True
    for split, split_stp_sz in [('train', stp_sz), ('valid', valid_stp_sz)]:
        split_files = _find_hdf_files(dataset, split)
        index = HDF5MmapIndex(split_files, all_signals)
        source = WindowedHDF5Source(index, win_sz, split_stp_sz, u, y)

        # Reference: compute expected window count using the same formula
        ref_total = 0
        for f in split_files:
            with h5py.File(f, 'r') as hf:
                seq_len = hf[u[0]].shape[0]
            n_win = max(0, (seq_len - win_sz) // split_stp_sz + 1)
            ref_total += n_win

        match = len(source) == ref_total
        print(f'  {split}: Grain={len(source)}, ref={ref_total}, match={match}')
        if not match:
            l2_pass = False

    all_passed &= l2_pass
    print(f'  LEVEL 2: {"PASS" if l2_pass else "FAIL"}')

    # ── Level 3: Raw windows (pre-normalization) ──
    print()
    print('=' * 60)
    print('Level 3: Raw windows (pre-normalization)')
    print('=' * 60)

    l3_pass = True
    for split, split_stp_sz in [('train', stp_sz), ('valid', valid_stp_sz)]:
        split_files = _find_hdf_files(dataset, split)
        index = HDF5MmapIndex(split_files, all_signals)
        source = WindowedHDF5Source(index, win_sz, split_stp_sz, u, y)

        n_samples = min(10, len(source))
        rng = np.random.default_rng(42)
        sample_indices = sorted(rng.choice(len(source), size=n_samples, replace=False))

        for idx in sample_indices:
            grain_item = source[idx]

            # Reference: resolve index and read with h5py directly
            import bisect
            file_idx = bisect.bisect_right(source.cum_windows, idx)
            prev = source.cum_windows[file_idx - 1] if file_idx > 0 else 0
            local_win = idx - prev
            l_slc = local_win * split_stp_sz
            r_slc = l_slc + win_sz
            path = source.file_paths[file_idx]

            with h5py.File(path, 'r') as hf:
                ref_u = np.stack([hf[s][l_slc:r_slc] for s in u], axis=-1).astype(np.float32)
                ref_y = np.stack([hf[s][l_slc:r_slc] for s in y], axis=-1).astype(np.float32)

            u_ok = np.allclose(grain_item['u'], ref_u, atol=1e-7)
            y_ok = np.allclose(grain_item['y'], ref_y, atol=1e-7)
            if not (u_ok and y_ok):
                print(f'  {split} idx={idx}: u_match={u_ok}, y_match={y_ok}')
                if not u_ok:
                    print(f'    max u diff: {np.max(np.abs(grain_item["u"] - ref_u))}')
                if not y_ok:
                    print(f'    max y diff: {np.max(np.abs(grain_item["y"] - ref_y))}')
                l3_pass = False

    all_passed &= l3_pass
    print(f'  LEVEL 3: {"PASS" if l3_pass else "FAIL"}')

    # ── Level 4: Raw validation batches ──
    print()
    print('=' * 60)
    print('Level 4: Raw validation batches')
    print('=' * 60)

    from .pipeline import create_grain_dls
    pipeline = create_grain_dls(u=u, y=y, dataset=dataset, win_sz=win_sz,
                                stp_sz=stp_sz, bs=64, seed=42)

    grain_valid_batches = list(pipeline.valid)

    # Reference: read all valid windows raw, batch manually
    valid_files = _find_hdf_files(dataset, 'valid')
    valid_index = HDF5MmapIndex(valid_files, all_signals)
    valid_source = WindowedHDF5Source(valid_index, win_sz, valid_stp_sz, u, y)

    all_items = [valid_source[i] for i in range(len(valid_source))]

    bs = 64
    manual_batches = []
    for b in range((len(all_items) + bs - 1) // bs):
        batch_items = all_items[b * bs: (b + 1) * bs]
        manual_batches.append({
            'u': np.stack([item['u'] for item in batch_items]),
            'y': np.stack([item['y'] for item in batch_items]),
        })

    l4_pass = True
    if len(grain_valid_batches) != len(manual_batches):
        print(f'  Batch count mismatch: Grain={len(grain_valid_batches)}, manual={len(manual_batches)}')
        l4_pass = False
    else:
        for i, (gb, mb) in enumerate(zip(grain_valid_batches, manual_batches)):
            u_ok = np.allclose(gb['u'], mb['u'], atol=1e-6)
            y_ok = np.allclose(gb['y'], mb['y'], atol=1e-6)
            if not (u_ok and y_ok):
                print(f'  Batch {i}: u_match={u_ok}, y_match={y_ok}')
                if not u_ok:
                    print(f'    max u diff: {np.max(np.abs(gb["u"] - mb["u"]))}')
                if not y_ok:
                    print(f'    max y diff: {np.max(np.abs(gb["y"] - mb["y"]))}')
                l4_pass = False

    all_passed &= l4_pass
    print(f'  LEVEL 4: {"PASS" if l4_pass else "FAIL"}')

    # ── Level 5: Test sequences (full-length, raw) ──
    print()
    print('=' * 60)
    print('Level 5: Test sequences (full-length)')
    print('=' * 60)

    grain_test_batches = list(pipeline.test)
    test_files = _find_hdf_files(dataset, 'test')

    l5_pass = True
    if len(grain_test_batches) != len(test_files):
        print(f'  Count mismatch: Grain={len(grain_test_batches)}, files={len(test_files)}')
        l5_pass = False
    else:
        for i, (gb, tf) in enumerate(zip(grain_test_batches, test_files)):
            with h5py.File(tf, 'r') as hf:
                raw_u = np.stack([hf[s][:] for s in u], axis=-1).astype(np.float32)
                raw_y = np.stack([hf[s][:] for s in y], axis=-1).astype(np.float32)

            grain_u = gb['u'][0]  # (1, seq_len, n_signals) -> (seq_len, n_signals)
            grain_y = gb['y'][0]

            u_ok = np.allclose(grain_u, raw_u, atol=1e-6)
            y_ok = np.allclose(grain_y, raw_y, atol=1e-6)
            if not (u_ok and y_ok):
                print(f'  File {i} ({Path(tf).name}): u_match={u_ok}, y_match={y_ok}')
                if not u_ok:
                    print(f'    max u diff: {np.max(np.abs(grain_u - raw_u))}')
                if not y_ok:
                    print(f'    max y diff: {np.max(np.abs(grain_y - raw_y))}')
                l5_pass = False

    all_passed &= l5_pass
    print(f'  LEVEL 5: {"PASS" if l5_pass else "FAIL"}')

    # ── Summary ──
    print()
    print('=' * 60)
    print(f'OVERALL: {"ALL PASSED" if all_passed else "SOME FAILED"}')
    print('=' * 60)

    return all_passed


def main():
    parser = argparse.ArgumentParser(description='Validate Grain vs TSFast data pipeline')
    parser.add_argument('--dataset', required=True, help='Path to dataset directory')
    parser.add_argument('--u', nargs='+', required=True, help='Input signal names')
    parser.add_argument('--y', nargs='+', required=True, help='Output signal names')
    parser.add_argument('--win_sz', type=int, default=100, help='Window size')
    parser.add_argument('--stp_sz', type=int, default=1, help='Step size')
    args = parser.parse_args()

    success = run_validation(
        dataset=args.dataset,
        u=args.u,
        y=args.y,
        win_sz=args.win_sz,
        stp_sz=args.stp_sz,
    )
    sys.exit(0 if success else 1)


if __name__ == '__main__':
    main()
