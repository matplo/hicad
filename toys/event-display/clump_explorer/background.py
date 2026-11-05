"""Pinned HDF5 catalog. No legacy HTTP decoding, clipping or display rounding."""
import hashlib
from pathlib import Path
import numpy as np
from .generator_adapter import core

MASSES = np.array([.13957039, .493677, core.PROTON_MASS])
NAMES = np.array([["pi+", "pi-"], ["K+", "K-"], ["p", "pbar"]])


def fingerprint(path):
    """Stat identity catches edits/replacement without hashing gigabytes per drag.

    Includes inode, size, mtime_ns and ctime_ns; intentionally not portable across
    copies. Recipes must be replayed against the pinned catalog instance/file.
    """
    try:
        s = path.stat()
    except OSError:
        raise core.ParameterError("background", "Pinned file is missing", "missing_background", 404)
    payload = f"{s.st_dev}:{s.st_ino}:{s.st_size}:{s.st_mtime_ns}:{s.st_ctime_ns}"
    return "stat-sha256:" + hashlib.sha256(payload.encode()).hexdigest()


def empty_measured():
    return {"particle_id": [], "species": [], "charge": [], "p4": [], "accepted": []}


def decode_rows(rows, event, observed=None):
    rows, event = np.asarray(rows, dtype=np.float64), np.asarray(event, dtype=np.float64)
    if rows.ndim != 2 or rows.shape[1] != 5 or not np.isfinite(rows).all() or not np.isfinite(event).all():
        core.fail("background", "Nonfinite or malformed active HDF5 rows")
    if len(rows) and (event[7] <= 0 or np.any(rows[:, 3] == 0)):
        core.fail("background", "Invalid mean pT or unsigned species mass code")
    index = np.argmin(abs(abs(rows[:, 3, None]) - np.log1p(MASSES)), axis=1)
    if np.any(abs(abs(rows[:, 3]) - np.log1p(MASSES[index])) > .002):
        core.fail("background", "Unknown signed species mass code")
    charge = np.where(rows[:, 3] > 0, 1, -1)
    species = NAMES[index, (charge < 0).astype(int)]
    if observed is not None:
        p4 = np.asarray(observed, dtype=np.float64)
        if p4.shape != (len(rows), 4) or not np.isfinite(p4).all() or np.any(p4[:, 0] <= 0):
            core.fail("background.observed/p4", "Expected finite aligned positive-energy [M,4] dataset")
        mass2 = p4[:, 0]**2 - np.sum(p4[:, 1:]**2, axis=1)
        # float32 high momentum subtraction needs a scale-aware tolerance.
        tolerance = 2e-6 * np.maximum(1, p4[:, 0]**2)
        if np.any(abs(mass2 - MASSES[index]**2) > tolerance):
            core.fail("background.observed/p4", "Mass shell disagrees with aligned species")
        k = core.kinematics(p4)
        encoded_pt = event[7] * np.exp(rows[:, 2])
        if not np.allclose(k["pt"], encoded_pt, rtol=.015, atol=1e-7) or not np.allclose(k["eta"], rows[:, 4]+event[9], atol=.015):
            core.fail("background.observed/p4", "Particle alignment disagrees with packed kinematics")
        delta_phi = (k["phi"]-np.arctan2(rows[:, 1], rows[:, 0])+np.pi) % (2*np.pi)-np.pi
        if np.any(abs(delta_phi) > .015):
            core.fail("background.observed/p4", "Particle alignment disagrees with packed azimuth")
    else:
        with np.errstate(over="ignore", invalid="ignore"):
            pt = event[7] * np.exp(rows[:, 2])
            eta = rows[:, 4] + event[9]
            phi = np.arctan2(rows[:, 1], rows[:, 0])
            px, py, pz = pt*np.cos(phi), pt*np.sin(phi), pt*np.sinh(eta)
            energy = np.sqrt(MASSES[index]**2 + px*px + py*py + pz*pz)
            p4 = np.column_stack((energy, px, py, pz))
        if not np.isfinite(p4).all():
            core.fail("background", "Decoded four-vectors are nonfinite")
    return p4, species.tolist(), charge.tolist()


class BackgroundProvider:
    def __init__(self, search_dirs=()):
        self.search_dirs = list(search_dirs)
        self.files, self.warnings = {}, []
        candidates = set()
        for item in self.search_dirs:
            p = Path(item).expanduser().resolve()
            if p.is_file():
                candidates.add(p)
            elif p.is_dir():
                candidates.update(p.glob("*.h5"))
                candidates.update(p.glob("output/*.h5"))
        if not candidates:
            return
        import h5py  # Source-only operation does not import or open HDF5.
        for path in sorted(candidates):
            try:
                fp = fingerprint(path)
                with h5py.File(path, "r") as f:
                    ev = np.asarray(f["event_record"][:], dtype=np.float64)
                    shape = f["point_cloud"].shape
                    mc = f["mc_record"][:]
                    if ev.ndim != 2 or ev.shape[1] != 10 or len(shape) != 3 or shape[0] != len(ev) or shape[2] != 5 or mc.shape != (2,):
                        raise ValueError("Incompatible dataset shapes")
                    if not np.isfinite(ev).all() or not np.isfinite(mc).all():
                        raise ValueError("Nonfinite metadata")
                    if np.any((ev[:, 0] < 0) | (ev[:, 0] > 1)) or np.any((ev[:, 1] < 0) | (ev[:, 1] > shape[1]) | (ev[:, 1] != np.floor(ev[:, 1]))):
                        raise ValueError("Invalid centrality or multiplicity")
                    observed = "observed/p4" in f
                    if observed and f["observed/p4"].shape != (*shape[:2], 4):
                        raise ValueError("Unaligned observed/p4 shape")
                    temperature = float(f.attrs["temperature_gev"]) if "temperature_gev" in f.attrs else None
                    if temperature is not None and (not np.isfinite(temperature) or temperature <= 0):
                        raise ValueError("Invalid recorded background temperature")
                    source_acceptance = "unknown"
                    if "acceptance_json" in f.attrs:
                        import json
                        source_acceptance = core.validate_acceptance(json.loads(f.attrs["acceptance_json"]))
                    physics = {"thermal_law": str(f.attrs.get("thermal_law", "unrecorded")),
                               "generator_version": str(f.attrs.get("generator_version", "unrecorded")),
                               "temperature_gev": temperature,
                               "temperature_policy": str(f.attrs.get("temperature_policy", "unrecorded")),
                               "multiplicity_policy": str(f.attrs.get("multiplicity_policy", "unrecorded")),
                               "boost_speed": float(f.attrs["background_boost_speed"]) if "background_boost_speed" in f.attrs else None}
                    if physics["boost_speed"] is not None and (not np.isfinite(physics["boost_speed"]) or not 0 <= physics["boost_speed"] < 1):
                        raise ValueError("Invalid recorded background boost speed")
                if fingerprint(path) != fp:
                    raise ValueError("File changed during catalog scan")
                dataset_id = "h5-" + hashlib.sha256(str(path).encode()).hexdigest()[:24]
                self.files[dataset_id] = {"path": path, "events": ev, "public": {
                    "dataset_id": dataset_id, "label": path.name, "event_count": len(ev),
                    "fingerprint": fp, "fingerprint_policy": "stat-sha256-v1", "point_cloud_shape": list(shape),
                    "mc_record": mc.tolist(), "observed_p4": observed, "source_acceptance": source_acceptance,
                    "physics": physics}}
            except (OSError, KeyError, ValueError) as exc:
                self.warnings.append(f"Skipped {path.name}: {exc}")

    def catalog(self):
        return [v["public"] for v in self.files.values()]

    def refresh(self):
        fresh = type(self)(self.search_dirs)
        # Entries are immutable after construction. Atomic replacement permits
        # requests already using an older entry to fail their fingerprint check.
        self.files, self.warnings = fresh.files, fresh.warnings

    def _entry(self, dataset_id):
        if dataset_id not in self.files:
            raise core.ParameterError("background.dataset_id", "Unknown dataset", "unknown_dataset", 404)
        return self.files[dataset_id]

    def check_reference(self, ref):
        if ref["provider"] == "none":
            return
        entry = self._entry(ref["dataset_id"])
        if ref["fingerprint"] != entry["public"]["fingerprint"] or fingerprint(entry["path"]) != ref["fingerprint"]:
            raise core.ParameterError("background.fingerprint", "Pinned file changed; refresh catalog and explicitly reselect", "changed_background", 409)
        if ref["event_index"] >= len(entry["events"]):
            raise core.ParameterError("background.event_index", "Unknown event", "unknown_event", 404)

    def select(self, request):
        keys = {"dataset_id", "mode", "event_index", "current_index", "centrality_min", "centrality_max", "selection_seed"}
        core.fields(request, keys, {"dataset_id", "mode", "centrality_min", "centrality_max", "selection_seed"}, "selection")
        if not isinstance(request["dataset_id"], str):
            core.fail("selection.dataset_id", "Expected public dataset ID")
        entry = self._entry(request["dataset_id"])
        mode = core.choice(request["mode"], ["exact", "random", "next", "previous"], "selection.mode")
        lo = core.number(request["centrality_min"], "selection.centrality_min", 0, 1)
        hi = core.number(request["centrality_max"], "selection.centrality_max", 0, 1)
        if lo >= hi:
            core.fail("selection.centrality_max", "Must exceed minimum")
        selection_seed = core.seed(request["selection_seed"], "selection.selection_seed")
        c = entry["events"][:, 0]
        matched = np.flatnonzero((c >= lo) & ((c < hi) | ((hi == 1) & (c == 1))))
        if not len(matched):
            raise core.ParameterError("selection.centrality_min", "No events match this window", "empty_selection", 404)
        if mode == "exact":
            idx = core.number(request.get("event_index"), "selection.event_index", 0, 2**53-1, True)
            if idx >= len(c) or idx not in matched:
                raise core.ParameterError("selection.event_index", "Event absent or outside selected window", "unknown_event", 404)
        elif mode == "random":
            idx = int(np.random.Generator(np.random.PCG64(int(selection_seed))).choice(matched))
        else:
            current = core.number(request.get("current_index"), "selection.current_index", 0, len(c)-1, True)
            eligible = matched[matched > current] if mode == "next" else matched[matched < current]
            idx = int(eligible[0 if mode == "next" else -1]) if len(eligible) else int(matched[0 if mode == "next" else -1])
        ref = {"provider": "hdf5", "dataset_id": request["dataset_id"], "event_index": idx,
               "fingerprint": entry["public"]["fingerprint"]}
        self.check_reference(ref)
        return {**ref, "selection": {**request, "selection_seed": selection_seed}}

    def read(self, ref, acceptance):
        if ref["provider"] == "none":
            return {"measured": empty_measured(), "metadata": {"event_record": None, "mc_record": None,
                    "geometry": None, "centrality": None, "source_acceptance": None}}
        self.check_reference(ref)
        entry = self._entry(ref["dataset_id"])
        import h5py
        with h5py.File(entry["path"], "r") as f:
            idx = ref["event_index"]
            ev = np.asarray(f["event_record"][idx], dtype=np.float64)
            m = int(ev[1])
            if m > core.LIMITS["particles"]:
                core.fail("background", "Supplied particle count exceeds 20000")
            rows = f["point_cloud"][idx, :m]
            observed = f["observed/p4"][idx, :m] if "observed/p4" in f else None
            p4, species, charge = decode_rows(rows, ev, observed)
            physics = dict(entry["public"]["physics"])
            if "truth/event/temperature_gev" in f:
                temperature = float(f["truth/event/temperature_gev"][idx])
                if not np.isfinite(temperature) or temperature <= 0:
                    core.fail("background.temperature_gev", "Invalid per-event temperature metadata")
                physics["temperature_gev"] = temperature
            emitted = int(f["truth/event/emitted_multiplicity"][idx]) if "truth/event/emitted_multiplicity" in f else None
            if emitted is not None and emitted < m:
                core.fail("background.emitted_multiplicity", "Emitted count smaller than stored accepted count")
        self.check_reference(ref)
        return {"measured": {"particle_id": [f"b:{ref['dataset_id']}:{idx}:{i}" for i in range(m)],
                "species": species, "charge": charge, "p4": p4.tolist(),
                "accepted": core.apply_acceptance(p4, acceptance).tolist()}, "metadata": {
                **entry["public"], "event_index": idx, "event_record": ev.tolist(),
                "physics": physics, "emitted_multiplicity": emitted,
                "centrality": float(ev[0]), "geometry": {"b": float(ev[2]), "npart": float(ev[3]), "ncoll": float(ev[4])},
                "precision": "observed_p4" if observed is not None else "decoded_float16",
                "acceptance_note": "Looser cuts cannot recover tracks absent from the stored background."}}
