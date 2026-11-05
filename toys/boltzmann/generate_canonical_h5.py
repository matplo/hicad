#!/usr/bin/env python3
"""Generate new HDF5 backgrounds with the clump's massive canonical 3D law.

The legacy generator/files remain untouched. Geometry and species abundances
reuse legacy energy configurations; momentum generation is shared with clumps.
The old multiplicity model now defines emitted counts, not accepted counts.
"""
import argparse
import json
import os
from pathlib import Path
import tempfile

import h5py
import numpy as np
import clumps
from canonical_background import BackgroundStreams, generate_background_event, SPECIES, MASS, CHARGE, VERSION, RNG_POLICY
from generate_thermal_h5 import CONFIGS, sample_glauber_and_multiplicity

EVENT_COLUMNS = ["centrality_class", "Multiplicity", "Impact_parameter", "N_part", "N_coll",
                 "sin(Psi2)", "cos(Psi2)", "Mean_pT", "Sum_pT", "Eta_mean"]
PARTICLE_COLUMNS = ["cos(phi)", "sin(phi)", "log(pT/mean_pT)", "Q * log(1 + mass)", "eta_i - eta_mean"]


def generate_dataset(setting, n_events=1000, temperature_gev=.1, root_seed="42",
                     output_dir=None, eta_max=1., pt_min_gev=0., pt_max_gev=None,
                     compression="lzf", overwrite=False, progress=None):
    clumps.choice(setting, list(CONFIGS), "setting")
    clumps.number(n_events, "n_events", 1, 1000000, True)
    clumps.number(temperature_gev, "temperature_gev", .001, .5)
    root_seed = clumps.seed(root_seed, "root_seed")
    acceptance = clumps.validate_acceptance({"eta_max": eta_max, "pt_min_gev": pt_min_gev,
                                            "pt_max_gev": pt_max_gev, "azimuth": "full"})
    clumps.choice(compression, ["lzf", "gzip"], "compression")
    cfg = CONFIGS[setting]
    directory = Path(output_dir) if output_dir is not None else Path(__file__).resolve().parent / "canonical-output"
    directory.mkdir(parents=True, exist_ok=True)
    path = directory / ("canonical_" + cfg.filename)
    if path.exists() and not overwrite:
        raise FileExistsError(f"{path} already exists; choose another output directory or explicitly use --overwrite")
    fd, temporary = tempfile.mkstemp(prefix=".canonical-", suffix=".h5.tmp", dir=directory)
    os.close(fd)
    try:
        with h5py.File(temporary, "w") as f:
            f.attrs.update({"setting_name": cfg.name, "Ecm_GeV": cfg.ecm, "mc_label": cfg.mc_label,
                "n_events": n_events, "generator_version": VERSION, "thermal_law": clumps.THERMAL_LAW,
                "core_version": clumps.CORE_VERSION, "rng_policy": RNG_POLICY, "root_seed": root_seed,
                "numpy_version": np.__version__, "temperature_gev": temperature_gev,
                "temperature_policy": "fixed_common_to_all_species_and_events",
                "background_boost_speed": 0., "acceptance_json": json.dumps(acceptance, sort_keys=True),
                "species_json": json.dumps(SPECIES), "species_probabilities_json": json.dumps(cfg.species_probs),
                "multiplicity_policy": "legacy_geometry_model_emitted_then_lab_acceptance_no_refill",
                "conservation_policy": "canonical_reservoir_emission_not_globally_conserving",
                "momentum_units": "GeV", "p4_columns": json.dumps(["E", "px", "py", "pz"])})
            mc = f.create_dataset("mc_record", data=np.array([cfg.ecm, cfg.mc_label], np.float32))
            mc.attrs["columns"] = ["Ecm", "Label"]
            ev = f.create_dataset("event_record", shape=(n_events, 10), dtype="f4", compression=compression)
            ev.attrs["columns"] = EVENT_COLUMNS
            # Grow capacity as needed; fill value keeps all inactive rows exactly zero.
            pc = f.create_dataset("point_cloud", shape=(n_events, 1, 5), maxshape=(n_events, None, 5),
                                  chunks=(1, 256, 5), dtype="f2", compression=compression, fillvalue=0)
            pc.attrs["columns"] = PARTICLE_COLUMNS
            obs = f.create_dataset("observed/p4", shape=(n_events, 1, 4), maxshape=(n_events, None, 4),
                                   chunks=(1, 256, 4), dtype="f4", compression=compression, fillvalue=0)
            obs.attrs["columns"] = ["E", "px", "py", "pz"]
            obs.attrs["units"] = "GeV"
            emitted_ds = f.create_dataset("truth/event/emitted_multiplicity", shape=(n_events,), dtype="i4")
            accepted_ds = f.create_dataset("truth/event/accepted_multiplicity", shape=(n_events,), dtype="i4")
            f.create_dataset("truth/event/temperature_gev", data=np.full(n_events, temperature_gev, np.float64))
            setting_id = int(round(cfg.mc_label * 100))
            maximum = 0
            for i in range(n_events):
                streams = BackgroundStreams(root_seed, i, setting_id)
                c, b, npart, ncoll, emitted = sample_glauber_and_multiplicity(cfg, 1, streams.rng(20))
                count = int(emitted[0])
                sample = generate_background_event(count, temperature_gev, cfg.species_probs,
                                                     acceptance=acceptance, stream_factory=streams)
                accepted_p4 = sample.accepted_p4
                indices = sample.species_index[sample.accepted]
                m = len(accepted_p4)
                maximum = max(maximum, m)
                if m > pc.shape[1]:
                    pc.resize((n_events, m, 5)); obs.resize((n_events, m, 4))
                k = clumps.kinematics(accepted_p4)
                row = np.zeros(10, np.float32)
                row[:5] = [c[0], m, b[0], npart[0], ncoll[0]]
                # These legacy columns remain compatible, with no invented flow plane.
                row[5:7] = [0., 1.]
                if m:
                    row[7:10] = [k["pt"].mean(), k["pt"].sum(), k["eta"].mean()]
                    # Normalize using the *declared float32 event means*.
                    features = np.column_stack((np.cos(k["phi"]), np.sin(k["phi"]),
                        np.log(k["pt"]/float(row[7])), CHARGE[indices]*np.log1p(MASS[indices]), k["eta"]-float(row[9])))
                    with np.errstate(over="ignore"):
                        features = features.astype(np.float16)
                    if not np.isfinite(features).all():
                        raise ValueError(f"Event {i} cannot be encoded with finite float16 features")
                    pc[i, :m] = features; obs[i, :m] = accepted_p4.astype(np.float32)
                ev[i] = row
                emitted_ds[i], accepted_ds[i] = count, m
                if progress is not None and ((i+1) % 100 == 0 or i+1 == n_events):
                    progress(f"[{setting}] {i+1}/{n_events} events")
            f.attrs["max_multiplicity"] = maximum
            f.attrs["event_plane_policy"] = "placeholder_only_no_anisotropic_flow"
        if overwrite:
            os.replace(temporary, path)
        else:
            # Publish atomically without replacing a file created concurrently.
            os.link(temporary, path)
        return path
    finally:
        Path(temporary).unlink(missing_ok=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--settings", nargs="+", choices=list(CONFIGS), default=list(CONFIGS))
    parser.add_argument("--n-events", type=int, default=1000)
    parser.add_argument("--temperature-gev", type=float, default=.1, help="Common canonical T for all species (default: 0.1 GeV)")
    parser.add_argument("--seed", default="42", help="Decimal uint64 root seed")
    parser.add_argument("--output-dir", type=Path, default=Path(__file__).resolve().parent / "canonical-output")
    parser.add_argument("--eta-max", type=float, default=1.)
    parser.add_argument("--pt-min-gev", type=float, default=0.)
    parser.add_argument("--pt-max-gev", type=float)
    parser.add_argument("--compression", choices=["lzf", "gzip"], default="lzf")
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()
    try:
        for setting in args.settings:
            path = generate_dataset(setting, args.n_events, args.temperature_gev, args.seed, args.output_dir,
                                    args.eta_max, args.pt_min_gev, args.pt_max_gev, args.compression,
                                    args.overwrite, lambda message: print(message, flush=True))
            print(f"Saved {path}", flush=True)
    except (ValueError, FileExistsError) as exc:
        parser.error(str(exc))


if __name__ == "__main__":
    main()
