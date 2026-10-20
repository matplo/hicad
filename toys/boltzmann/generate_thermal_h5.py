#!/usr/bin/env python3
"""
Generate HDF5 files containing 100,000 vanilla thermal heavy-ion collision events
for three collision energy settings:
  1. Top LHC (Pb-Pb at sqrt(s_NN) = 5020 GeV, MC label = 0.01)
  2. Top RHIC (Au-Au at sqrt(s_NN) = 200 GeV, MC label = 0.02)
  3. RHIC BES (Au-Au at sqrt(s_NN) = 9.2 GeV, MC label = 0.03)

HDF5 Schema per file:
  - mc_record: float32, shape (2,)
      [0] Ecm (sqrt(s_NN) in GeV)
      [1] Label (MC model identifier: 0.01, 0.02, 0.03 for vanilla thermal sets)
  - event_record: float32, shape (N_events, 10)
      [0] centrality class (0-1, 0 = most central, 1 = 100% peripheral)
      [1] Multiplicity (number of particles in event)
      [2] Impact parameter b (fm)
      [3] Number of participants (N_part)
      [4] Number of binary nucleon collisions (N_coll)
      [5] Event plane sin(Psi2)
      [6] Event plane cos(Psi2)
      [7] Mean pT (GeV/c, calculated from particles)
      [8] Sum pT (GeV/c, calculated from particles)
      [9] Eta_mean (calculated from particles)
  - point_cloud: float16, shape (N_events, max_M, 5), zero-padded for i >= Multiplicity
      [0] cos(phi)
      [1] sin(phi)
      [2] log(pT / mean_pT)
      [3] Q * log(1 + mass)
      [4] eta_i / eta_mean
"""

from __future__ import annotations

import argparse
import os
import time
from dataclasses import dataclass
from typing import Dict, Tuple

import h5py
import numpy as np


# Charged hadron species: (name, mass in GeV/c^2, charge Q in units of e)
SPECIES = [
    ("pi+", 0.13957039, +1.0),
    ("pi-", 0.13957039, -1.0),
    ("K+", 0.493677, +1.0),
    ("K-", 0.493677, -1.0),
    ("p", 0.938272088, +1.0),
    ("pbar", 0.938272088, -1.0),
]

SPECIES_MASS = np.array([s[1] for s in SPECIES], dtype=np.float32)
SPECIES_CHARGE = np.array([s[2] for s in SPECIES], dtype=np.float32)
# Precompute Q * log(1 + mass) for each species
SPECIES_QLOGM = (SPECIES_CHARGE * np.log(1.0 + SPECIES_MASS)).astype(np.float16)


@dataclass
class EnergyConfig:
    name: str
    filename: str
    ecm: float
    mc_label: float
    nucleus_A: int
    b_max: float
    npart_max: float
    ncoll_max: float
    x_hard: float
    mult_central_0_5: float
    mean_pt_central: float
    mean_pt_peripheral: float
    species_probs: Tuple[float, float, float, float, float, float]
    eta_max: float = 1.0


CONFIGS: Dict[str, EnergyConfig] = {
    "lhc": EnergyConfig(
        name="Top LHC (Pb-Pb 5.02 TeV)",
        filename="thermal_events_lhc_5020GeV.h5",
        ecm=5020.0,
        mc_label=0.01,
        nucleus_A=208,
        b_max=15.6,
        npart_max=406.0,
        ncoll_max=2050.0,
        x_hard=0.13,
        mult_central_0_5=1940.0,
        mean_pt_central=0.68,
        mean_pt_peripheral=0.56,
        species_probs=(0.415, 0.415, 0.062, 0.062, 0.023, 0.023),
    ),
    "rhic": EnergyConfig(
        name="Top RHIC (Au-Au 200 GeV)",
        filename="thermal_events_rhic_200GeV.h5",
        ecm=200.0,
        mc_label=0.02,
        nucleus_A=197,
        b_max=14.5,
        npart_max=382.0,
        ncoll_max=1065.0,
        x_hard=0.11,
        mult_central_0_5=680.0,
        mean_pt_central=0.56,
        mean_pt_peripheral=0.46,
        species_probs=(0.405, 0.405, 0.065, 0.065, 0.033, 0.027),
    ),
    "bes": EnergyConfig(
        name="RHIC BES (Au-Au 9.2 GeV)",
        filename="thermal_events_bes_9p2GeV.h5",
        ecm=9.2,
        mc_label=0.03,
        nucleus_A=197,
        b_max=14.0,
        npart_max=355.0,
        ncoll_max=750.0,
        x_hard=0.06,
        mult_central_0_5=80.0,
        mean_pt_central=0.42,
        mean_pt_peripheral=0.35,
        species_probs=(0.360, 0.380, 0.080, 0.040, 0.135, 0.005),
    ),
}


def sample_glauber_and_multiplicity(
    cfg: EnergyConfig, n_events: int, rng: np.random.Generator
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Sample minimum-bias centrality in [0, 1] and corresponding Glauber quantities:
      - centrality: float32 in [0, 1] (0 = central, 1 = peripheral)
      - b: impact parameter (fm)
      - n_part: number of participating nucleons
      - n_coll: number of binary nucleon-nucleon collisions
      - mult: integer charged particle multiplicity (>= 2)
    """
    centrality = rng.uniform(0.0, 1.0, size=n_events).astype(np.float32)

    # Impact parameter b ~ b_max * sqrt(c) with small geometric fluctuation
    b_mean = cfg.b_max * np.sqrt(centrality)
    b = np.clip(
        b_mean + rng.normal(0.0, 0.12, size=n_events), 0.0, cfg.b_max * 1.08
    ).astype(np.float32)

    # Smooth Glauber parameterization of N_part(c) and N_coll(c)
    # Calibrated to match Woods-Saxon Glauber curves across centrality 0 -> 1
    c_safe = np.clip(centrality, 0.0, 0.9999)
    npart_mean = (
        cfg.npart_max
        * np.power(1.0 - np.power(c_safe, 0.85), 1.65)
        * np.exp(-0.55 * c_safe)
    )
    npart_mean = np.maximum(npart_mean, 2.0)
    # Add event-by-event nucleon position fluctuation
    npart_sigma = np.sqrt(np.maximum(npart_mean * 0.6, 1.0))
    n_part = np.clip(
        np.round(rng.normal(npart_mean, npart_sigma)),
        2.0,
        float(2 * cfg.nucleus_A),
    ).astype(np.float32)

    # Binary collisions scale as (N_part / N_part_max)^1.33
    ncoll_mean = cfg.ncoll_max * np.power(n_part / cfg.npart_max, 1.33)
    ncoll_sigma = np.sqrt(np.maximum(ncoll_mean * 1.2, 1.0))
    n_coll = np.clip(
        np.round(rng.normal(ncoll_mean, ncoll_sigma)),
        1.0,
        cfg.ncoll_max * 1.15,
    ).astype(np.float32)

    # Two-component ancestor scaling for multiplicity:
    # N_anc(c) = (1 - x_hard) * (N_part / 2) + x_hard * N_coll
    anc = (1.0 - cfg.x_hard) * (n_part * 0.5) + cfg.x_hard * n_coll
    # Compute reference ancestor count averaged over 0-5% central window
    c_grid = np.linspace(0.0, 0.05, 500)
    npart_grid = (
        cfg.npart_max
        * np.power(1.0 - np.power(c_grid, 0.85), 1.65)
        * np.exp(-0.55 * c_grid)
    )
    ncoll_grid = cfg.ncoll_max * np.power(npart_grid / cfg.npart_max, 1.33)
    anc_ref = float(
        np.mean((1.0 - cfg.x_hard) * (npart_grid * 0.5) + cfg.x_hard * ncoll_grid)
    )

    mult_mean = cfg.mult_central_0_5 * (anc / anc_ref)
    # Glauber + NBD convolution: each ancestor has k_anc ~ 2.0, so total shape k_total = k_anc * N_anc
    k_nbd = np.maximum(2.0 * anc, 2.0)
    gamma_scale = np.maximum(mult_mean / k_nbd, 1e-3)
    lam = rng.gamma(shape=k_nbd, scale=gamma_scale, size=n_events)
    mult = np.maximum(rng.poisson(lam), 2).astype(np.int32)

    return centrality, b, n_part, n_coll, mult


def generate_dataset_for_config(
    cfg: EnergyConfig,
    output_dir: str,
    n_events: int = 100_000,
    batch_size: int = 5_000,
    seed: int = 42,
    compression: str = "lzf",
) -> str:
    """
    Generate the HDF5 file for a single EnergyConfig in batches to keep RAM usage low
    and write speed high.
    """
    t0 = time.time()
    rng = np.random.default_rng(seed)

    centrality, b, n_part, n_coll, mult = sample_glauber_and_multiplicity(
        cfg, n_events, rng
    )
    max_m = int(mult.max())

    # Event plane angle Psi2 ~ Uniform(-pi, pi)
    psi2 = rng.uniform(-np.pi, np.pi, size=n_events).astype(np.float32)
    sin_psi2 = np.sin(psi2).astype(np.float32)
    cos_psi2 = np.cos(psi2).astype(np.float32)

    # Centrality-dependent target <pT>(c):
    # Stronger radial flow in central events -> higher <pT>, calibrated so 0-5% mean = mean_pt_central
    c_flow_raw = np.power(1.0 - np.clip(centrality, 0.0, 1.0), 0.45)
    c_flow_0_5_mean = float(np.mean(np.power(1.0 - np.linspace(0.0, 0.05, 500), 0.45)))
    c_flow_factor = c_flow_raw / c_flow_0_5_mean
    target_mean_pt_c = (
        cfg.mean_pt_peripheral
        + (cfg.mean_pt_central - cfg.mean_pt_peripheral) * c_flow_factor
    ).astype(np.float32)

    # Relative species effective temperature weight: heavier particles get a radial-flow boost
    # In thermal + radial flow blast-wave: <pT>_s ~ T_kin + alpha_flow * mass_s
    probs = np.array(cfg.species_probs, dtype=np.float64)
    probs /= probs.sum()
    species_rel_weight = 1.0 + 0.65 * (SPECIES_MASS - 0.13957039)
    avg_rel_weight = float(np.sum(probs * species_rel_weight))
    species_norm_weight = (species_rel_weight / avg_rel_weight).astype(np.float32)

    os.makedirs(output_dir, exist_ok=True)
    out_path = os.path.join(output_dir, cfg.filename)

    event_record = np.zeros((n_events, 10), dtype=np.float32)
    event_record[:, 0] = centrality
    event_record[:, 1] = mult.astype(np.float32)
    event_record[:, 2] = b
    event_record[:, 3] = n_part
    event_record[:, 4] = n_coll
    event_record[:, 5] = sin_psi2
    event_record[:, 6] = cos_psi2

    central_0_5 = centrality < 0.05
    central_mult_avg = float(mult[central_0_5].mean()) if np.any(central_0_5) else 0.0
    print(
        f"[{cfg.name}] Generating {n_events:,} events "
        f"(max_M={max_m}, <M>_0-5%={central_mult_avg:.1f}, "
        f"<M>_MB={mult.mean():.1f}) -> {out_path}"
    )

    with h5py.File(out_path, "w") as h5f:
        # 1. MC record: one entry per file - float32: [Ecm, Label]
        mc_data = np.array([cfg.ecm, cfg.mc_label], dtype=np.float32)
        mc_ds = h5f.create_dataset("mc_record", data=mc_data, dtype=np.float32)
        mc_ds.attrs["columns"] = ["Ecm", "Label"]
        mc_ds.attrs["Ecm_GeV"] = float(cfg.ecm)
        mc_ds.attrs["Label"] = float(cfg.mc_label)
        mc_ds.attrs["model_description"] = (
            "0.0X: vanilla thermal events (Rita); "
            "0.1X: UrQMD; 0.2X: Angantyr; 0.3X: JEWEL; 0.4X: HIJING; 0.5X: AMPT"
        )

        # 2. Point Cloud Event Record: float16, shape (n_events, max_m, 5)
        chunk_events = min(
            n_events, batch_size, max(100, 500_000 // max(max_m, 1))
        )
        pc_ds = h5f.create_dataset(
            "point_cloud",
            shape=(n_events, max_m, 5),
            dtype=np.float16,
            chunks=(chunk_events, max_m, 5),
            compression=compression,
        )
        pc_ds.attrs["columns"] = [
            "cos(phi)",
            "sin(phi)",
            "log(pT/mean_pT)",
            "Q * log(1 + mass)",
            "eta_i / eta_mean",
        ]

        # Generate particles batch by batch in vectorized fashion
        f16_max = 65500.0
        for start in range(0, n_events, batch_size):
            end = min(start + batch_size, n_events)
            b_n = end - start
            b_mult = mult[start:end]
            b_max_m = int(b_mult.max())
            total_p = int(b_mult.sum())

            # Map each flat particle index back to its event index in [0, b_n)
            ev_idx = np.repeat(np.arange(b_n, dtype=np.int32), b_mult)

            # Local particle slot index within each event: 0 .. M_e - 1
            offsets = np.zeros(b_n, dtype=np.int64)
            if b_n > 1:
                np.cumsum(b_mult[:-1], out=offsets[1:])
            slot_idx = np.arange(total_p, dtype=np.int64) - np.repeat(offsets, b_mult)

            # Sample species for all particles in batch
            sp_idx = rng.choice(len(SPECIES), size=total_p, p=probs)
            q_log_m = SPECIES_QLOGM[sp_idx]

            # Sample thermal pT from Gamma(shape=2, scale=T_eff):
            # For Gamma(k=2, theta), E[pT] = 2 * theta, so theta = <pT>_target / 2
            ev_mean_pt_target = target_mean_pt_c[start:end][ev_idx]
            pt_mean_particle = ev_mean_pt_target * species_norm_weight[sp_idx]
            pt = rng.gamma(
                shape=2.0, scale=pt_mean_particle * 0.5, size=total_p
            ).astype(np.float32)
            pt = np.maximum(pt, 1e-4)

            # Sample azimuth phi ~ Uniform(-pi, pi) (vanilla thermal isotropic emission)
            phi = rng.uniform(-np.pi, np.pi, size=total_p).astype(np.float32)
            cos_phi = np.cos(phi).astype(np.float16)
            sin_phi = np.sin(phi).astype(np.float16)

            # Sample pseudorapidity eta ~ Uniform(-eta_max, eta_max)
            eta = rng.uniform(-cfg.eta_max, cfg.eta_max, size=total_p).astype(
                np.float32
            )

            # Compute exact per-event sums and means from generated particles
            sum_pt = np.bincount(ev_idx, weights=pt, minlength=b_n).astype(np.float32)
            mean_pt = (sum_pt / b_mult).astype(np.float32)

            sum_eta = np.bincount(ev_idx, weights=eta, minlength=b_n).astype(np.float32)
            eta_mean = (sum_eta / b_mult).astype(np.float32)

            event_record[start:end, 7] = mean_pt
            event_record[start:end, 8] = sum_pt
            event_record[start:end, 9] = eta_mean

            # Compute particle features relative to event averages
            log_pt_ratio = np.log(pt / mean_pt[ev_idx])
            log_pt_ratio_f16 = np.clip(log_pt_ratio, -f16_max, f16_max).astype(
                np.float16
            )

            # Guard against tiny eta_mean denominator so float16 never overflows to Inf
            eta_denom = eta_mean[ev_idx]
            eta_denom_safe = np.where(
                np.abs(eta_denom) < 1e-4,
                np.where(eta_denom >= 0.0, 1e-4, -1e-4),
                eta_denom,
            )
            eta_ratio = eta / eta_denom_safe
            eta_ratio_f16 = np.clip(eta_ratio, -f16_max, f16_max).astype(np.float16)

            # Assemble 3D padded batch array
            pc_batch = np.zeros((b_n, max_m, 5), dtype=np.float16)
            pc_batch[ev_idx, slot_idx, 0] = cos_phi
            pc_batch[ev_idx, slot_idx, 1] = sin_phi
            pc_batch[ev_idx, slot_idx, 2] = log_pt_ratio_f16
            pc_batch[ev_idx, slot_idx, 3] = q_log_m
            pc_batch[ev_idx, slot_idx, 4] = eta_ratio_f16

            pc_ds[start:end, :, :] = pc_batch

            if (end % 25_000 == 0) or (end == n_events):
                elapsed = time.time() - t0
                print(
                    f"  [{cfg.name}] Wrote {end:,}/{n_events:,} events ({elapsed:.1f}s)"
                )

        # 3. Event Record: float32, shape (n_events, 10)
        ev_ds = h5f.create_dataset(
            "event_record",
            data=event_record,
            dtype=np.float32,
            compression=compression,
        )
        ev_ds.attrs["columns"] = [
            "centrality_class",
            "Multiplicity",
            "Impact_parameter",
            "N_part",
            "N_coll",
            "sin(Psi2)",
            "cos(Psi2)",
            "Mean_pT",
            "Sum_pT",
            "Eta_mean",
        ]

        # Root metadata attributes
        h5f.attrs["setting_name"] = cfg.name
        h5f.attrs["Ecm_GeV"] = float(cfg.ecm)
        h5f.attrs["mc_label"] = float(cfg.mc_label)
        h5f.attrs["n_events"] = int(n_events)
        h5f.attrs["max_multiplicity"] = int(max_m)

    total_time = time.time() - t0
    file_mb = os.path.getsize(out_path) / (1024 * 1024)
    central_mask = event_record[:, 0] < 0.05
    print(
        f"[{cfg.name}] Done in {total_time:.1f}s | Size: {file_mb:.1f} MB | "
        f"Central (0-5%) <M>={event_record[central_mask, 1].mean():.1f}, "
        f"<pT>={event_record[central_mask, 7].mean():.4f} GeV/c\n"
    )
    return out_path


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Generate heavy-ion thermal event HDF5 files."
    )
    parser.add_argument(
        "--settings",
        nargs="+",
        choices=["lhc", "rhic", "bes"],
        default=["lhc", "rhic", "bes"],
        help="Which energy settings to generate.",
    )
    parser.add_argument(
        "--n-events",
        type=int,
        default=100_000,
        help="Number of events per file (default: 100,000).",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=5_000,
        help="Number of events generated per batch.",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="output",
        help="Directory to write the .h5 files.",
    )
    parser.add_argument(
        "--compression",
        type=str,
        default="lzf",
        choices=["lzf", "gzip"],
        help="HDF5 compression filter (default: lzf).",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Base random seed.",
    )
    args = parser.parse_args()

    for idx, key in enumerate(args.settings):
        cfg = CONFIGS[key]
        generate_dataset_for_config(
            cfg=cfg,
            output_dir=args.output_dir,
            n_events=args.n_events,
            batch_size=args.batch_size,
            seed=args.seed + idx * 100,
            compression=args.compression,
        )


if __name__ == "__main__":
    main()
