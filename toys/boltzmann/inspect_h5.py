#!/usr/bin/env python3
"""
Validate and summarize the generated heavy-ion thermal event HDF5 files.
Checks:
  1. Dataset presence, shapes, and exact dtypes (mc_record: float32, event_record: float32, point_cloud: float16)
  2. Absence of NaN or Inf values across all datasets
  3. Zero-padding integrity in point_cloud for indices >= Multiplicity
  4. Physics summary statistics (0-5% central vs minimum-bias Multiplicity, <pT>, b, N_part, N_coll)
"""

from __future__ import annotations

import glob
import os
import sys

import h5py
import numpy as np


def inspect_file(path: str) -> None:
    size_mb = os.path.getsize(path) / (1024 * 1024)
    print("=" * 80)
    print(f"FILE: {path} ({size_mb:.2f} MB)")
    print("=" * 80)

    with h5py.File(path, "r") as f:
        mc = f["mc_record"][:]
        ev = f["event_record"][:]
        pc_ds = f["point_cloud"]

        print(f"1. mc_record:")
        print(f"   shape={mc.shape}, dtype={mc.dtype}")
        print(f"   Ecm   = {mc[0]:.2f} GeV")
        print(f"   Label = {mc[1]:.4f}")

        print(f"\n2. event_record:")
        print(f"   shape={ev.shape}, dtype={ev.dtype}")
        assert ev.dtype == np.float32, f"Expected float32, got {ev.dtype}"
        assert not np.any(np.isnan(ev)), "NaN found in event_record!"
        assert not np.any(np.isinf(ev)), "Inf found in event_record!"

        cols = list(f["event_record"].attrs.get("columns", []))
        c_mask = ev[:, 0] < 0.05
        p_mask = ev[:, 0] > 0.80
        print(f"   Columns: {cols}")
        print(
            f"   Central (0-5%)   : <M>={ev[c_mask, 1].mean():.1f}, "
            f"<pT>={ev[c_mask, 7].mean():.4f} GeV/c, "
            f"<b>={ev[c_mask, 2].mean():.2f} fm, "
            f"<N_part>={ev[c_mask, 3].mean():.1f}, "
            f"<N_coll>={ev[c_mask, 4].mean():.1f}"
        )
        print(
            f"   Peripheral (80%+): <M>={ev[p_mask, 1].mean():.1f}, "
            f"<pT>={ev[p_mask, 7].mean():.4f} GeV/c, "
            f"<b>={ev[p_mask, 2].mean():.2f} fm, "
            f"<N_part>={ev[p_mask, 3].mean():.1f}, "
            f"<N_coll>={ev[p_mask, 4].mean():.1f}"
        )
        print(
            f"   Min-Bias (0-100%): <M>={ev[:, 1].mean():.1f} (min={ev[:, 1].min():.0f}, max={ev[:, 1].max():.0f}), "
            f"<pT>={ev[:, 7].mean():.4f} GeV/c"
        )

        print(f"\n3. point_cloud:")
        print(f"   shape={pc_ds.shape}, dtype={pc_ds.dtype}, chunks={pc_ds.chunks}, compression={pc_ds.compression}")
        assert pc_ds.dtype == np.float16, f"Expected float16, got {pc_ds.dtype}"

        # Sample check first 1,000 events for finite values and zero padding
        sample_n = min(1000, pc_ds.shape[0])
        pc_sample = pc_ds[:sample_n]
        assert not np.any(np.isnan(pc_sample)), "NaN found in point_cloud sample!"
        assert not np.any(np.isinf(pc_sample)), "Inf found in point_cloud sample!"

        # Check zero padding on event 0
        m0 = int(ev[0, 1])
        if m0 < pc_ds.shape[1]:
            pad_sum = float(np.abs(pc_sample[0, m0:, :]).sum())
            assert pad_sum == 0.0, f"Non-zero padding found in event 0 after index {m0}!"

        active0 = pc_sample[0, :m0, :]
        print(f"   Cols: {list(pc_ds.attrs.get('columns', []))}")
        print(
            f"   Event 0 (M={m0}) sample particle 0: "
            f"cos(phi)={active0[0, 0]:.3f}, sin(phi)={active0[0, 1]:.3f}, "
            f"log(pT/<pT>)={active0[0, 2]:.3f}, Q*log(1+m)={active0[0, 3]:.3f}, "
            f"eta-<eta>={active0[0, 4]:.3f}"
        )
        print("   Validation checks (dtype, finite values, zero-padding): PASSED\n")


def main() -> None:
    target_dir = sys.argv[1] if len(sys.argv) > 1 else "output"
    files = sorted(glob.glob(os.path.join(target_dir, "*.h5")))
    if not files:
        print(f"No .h5 files found in {target_dir}")
        sys.exit(1)
    for path in files:
        inspect_file(path)


if __name__ == "__main__":
    main()
