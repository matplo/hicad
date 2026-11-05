"""Pure scenario composition, observation precision and stable aligned ordering."""
from copy import deepcopy
from dataclasses import dataclass, field
import hashlib
import json
import time
import threading
import numpy as np
from .generator_adapter import core
from .contracts import ScenarioSpec, COMPOSITIONS, SCHEMA
from .background import BackgroundProvider, empty_measured, MASSES, decode_rows


def canonical(value):
    return json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False)


def scenario_id(spec):
    identity = {"physical": spec.physical(), "core": core.CORE_VERSION,
                "rng": core.RNG_POLICY, "numpy": np.__version__}
    if "event_mixed" in spec.request["clump_config"]["variants"]:
        identity["event_mixing_policy"] = core.EVENT_MIXING_POLICY
    return hashlib.sha256(canonical(identity).encode()).hexdigest()[:32]


def select_rows(measured, indices):
    return {k: [v[i] for i in indices] for k, v in measured.items()}


def summary(measured):
    p4 = np.asarray(measured["p4"], dtype=np.float64).reshape(-1, 4)
    k = core.kinematics(p4)
    n = len(p4)
    return {"count": n, "protons": measured["species"].count("p"), "antiprotons": measured["species"].count("pbar"),
            "sum_pt_gev": float(k["pt"].sum()), "mean_pt_gev": float(k["pt"].mean()) if n else None,
            "mean_eta": float(k["eta"].mean()) if n else None}


def packed_view(measured):
    """Recompute means for this composition, encode float16 and decode float32 means.

    Acceptance is the canonical lab mask. Quantization is an observation transform,
    not a second selection step. Preserve canonical particle IDs and species.
    """
    if not measured["p4"]:
        return deepcopy(measured)
    p4 = np.asarray(measured["p4"], dtype=np.float64)
    k = core.kinematics(p4)
    species = measured["species"]
    mass_map = {"pi+": MASSES[0], "pi-": MASSES[0], "K+": MASSES[1], "K-": MASSES[1], "p": MASSES[2], "pbar": MASSES[2]}
    means = [float(k["pt"].mean()), float(k["eta"].mean())]
    features = np.column_stack((np.cos(k["phi"]), np.sin(k["phi"]), np.log(k["pt"]/means[0]),
                                np.asarray(measured["charge"])*np.log1p([mass_map[s] for s in species]), k["eta"]-means[1]))
    with np.errstate(over="ignore"):
        features = features.astype(np.float16)
    if not np.isfinite(features).all():
        core.fail("observation_mode", "Composition cannot be represented in packed_float16")
    event = np.zeros(10, dtype=np.float32)
    event[1], event[7], event[8], event[9] = len(p4), means[0], k["pt"].sum(), means[1]
    decoded, decoded_species, _ = decode_rows(features, event)
    if decoded_species != species:
        core.fail("observation_mode", "Packed species alignment failed")
    out = deepcopy(measured)
    out["p4"] = decoded.tolist()
    return out


@dataclass
class ScenarioBundle:
    data: dict
    # Derived views/diagnostics are separate caches, never overwrite canonical p4.
    observation_cache: dict = field(default_factory=dict)
    diagnostic_cache: dict = field(default_factory=dict)
    lock: object = field(default_factory=threading.RLock, repr=False)

    def observation(self, variant, composition, mode="native"):
        key = (mode, variant, composition)
        if key in self.observation_cache:
            return self.observation_cache[key]
        components = []
        if composition in ("background_only", "combined"):
            components.append(self.data["background"]["measured"])
        if composition in ("source_only", "combined"):
            components.append(self.data["variants"][variant]["measured"])
        out = empty_measured()
        for component in components:
            idx = [i for i, accepted in enumerate(component["accepted"]) if accepted]
            selected = select_rows(component, idx)
            for name in out:
                out[name].extend(selected[name])
        order = self.data.get("observation_order_by_variant", {}).get(variant, self.data["observation_order"])[composition]
        ranks = {pid: i for i, pid in enumerate(order)}
        out = select_rows(out, sorted(range(len(out["p4"])), key=lambda i: ranks[out["particle_id"][i]]))
        if mode == "packed_float16":
            out = packed_view(out)
        self.observation_cache[key] = out
        return out

    def response(self, spec):
        result = deepcopy(self.data)
        result["request_id"] = spec.request["request_id"]
        result["resolved_request"] = deepcopy(spec.request)
        mode = spec.request["observation_mode"]
        result["summaries"], result["diagnostics"] = {}, {}
        if mode == "packed_float16":
            result["observation_views"] = {}
        for variant in result["variants"]:
            views = {c: self.observation(variant, c, mode) for c in COMPOSITIONS}
            result["summaries"][variant] = {c: summary(v) for c, v in views.items()}
            if mode == "packed_float16":
                result["observation_views"][variant] = views
        result["diagnostics_status"] = "deferred" if spec.request["diagnostics"]["enabled"] else "disabled"
        result["provenance"]["observation_mode"] = mode
        return result


def generate_scenario(spec, background_provider=None):
    """The same deterministic entry point for explorer and future batch code.

    No HTTP operations or persistent writes. A provider may perform a controlled
    read of the explicitly pinned background.
    """
    start = time.perf_counter()
    spec = ScenarioSpec.parse(spec)
    request = spec.request
    provider = background_provider if background_provider is not None else BackgroundProvider()
    config, realization = request["clump_config"], request["realization"]
    bg = provider.read(request["background"], config["acceptance"])
    count = config["clumps_per_injected_event"]["value"] if config["enabled"] else 0
    samples = [core.generate_clump(config, realization, i) for i in range(count)]
    p4 = np.concatenate([s.lab_p4 for s in samples]) if samples else np.empty((0, 4))
    rest = np.concatenate([s.rest_p4 for s in samples]) if samples else np.empty((0, 4))
    ids = sum([s.particle_id for s in samples], [])
    clump_ids = [i for i, s in enumerate(samples) for _ in s.particle_id]
    masks = np.concatenate([s.accepted for s in samples]).tolist() if samples else []
    if len(p4) + len(bg["measured"]["p4"]) > core.LIMITS["particles"]:
        core.fail("clump_config", "Background plus emitted daughters exceeds 20000 supplied particles")
    source = {"particle_id": ids, "species": ["p"]*len(ids), "charge": [1]*len(ids),
              "p4": p4.tolist(), "accepted": masks}
    truth = {"clump_id": clump_ids, "rest_p4": rest.tolist(), "beta_vectors": [s.realized["beta_vector"] for s in samples]}
    streams = core.StreamFactory(realization)
    decohered = p4.copy()
    pt = np.hypot(p4[:, 1], p4[:, 2])
    for i, pid in enumerate(ids):
        daughter = int(pid.split(":d")[1])
        phi = streams.rng(7, clump_ids[i], daughter).uniform(-np.pi, np.pi)
        decohered[i, 1:3] = pt[i] * np.array([np.cos(phi), np.sin(phi)])
    variants = {}
    for v in config["variants"]:
        if v == "event_mixed":
            mixed = [core.generate_event_mixed_clump(config, realization, i) for i in range(count)]
            mixed_p4 = np.concatenate([s.lab_p4 for s in mixed]) if mixed else np.empty((0, 4))
            mixed_rest = np.concatenate([s.rest_p4 for s in mixed]) if mixed else np.empty((0, 4))
            mixed_ids = sum([s.particle_id for s in mixed], [])
            mixed_mask = np.concatenate([s.accepted for s in mixed]).tolist() if mixed else []
            mixed_truth = {k: sum([s.donors[k] for s in mixed], []) for k in
                           ("donor_event_id", "temperature_gev", "boost_speed", "beta_vectors", "slot_clump_id")}
            mixed_truth.update(clump_id=[-1]*len(mixed_ids), rest_p4=mixed_rest.tolist(),
                               boost_vector_scope="particle", rest_frame_scope="individual_donor", mixing_policy=core.EVENT_MIXING_POLICY)
            variants[v] = {"measured": {"particle_id": mixed_ids, "species": ["p"]*len(mixed_ids),
                           "charge": [1]*len(mixed_ids), "p4": mixed_p4.tolist(), "accepted": mixed_mask},
                           "truth": mixed_truth, "source_truth": [s.realized for s in mixed],
                           "control_note": "Independent donor sources; ensemble one-body reference, not an event-wise kinematic twin. Accepted counts may differ."}
            continue
        measured = deepcopy(source)
        if v == "azimuth_decohered":
            measured["p4"] = decohered.tolist()
        variants[v] = {"measured": measured, "truth": deepcopy(truth), "source_truth": [s.realized for s in samples]}
    # Stable aligned ordering for exact twins; distinct slots for independent donors.
    ranks = {pid: float(streams.rng(8, clump_ids[i], int(pid.split(":d")[1])).random()) for i, pid in enumerate(ids)}
    ranks.update({pid: float(streams.rng(8, 8, i).random()) for i, pid in enumerate(bg["measured"]["particle_id"])})
    if "event_mixed" in variants:
        for pid, slot in zip(variants["event_mixed"]["measured"]["particle_id"], variants["event_mixed"]["truth"]["slot_clump_id"]):
            ranks[pid] = float(core.EventMixedStreams(realization, slot, int(pid.split(":d")[1])).rng(8).random())
    def shuffled(particle_ids):
        if not config["shuffle_particles"]:
            # Explicit opt-out is recipe metadata, never the default.
            return particle_ids
        return sorted(particle_ids, key=lambda pid: ranks[pid])
    source_ids = [pid for pid, passed in zip(ids, masks) if passed]
    background_ids = [pid for pid, passed in zip(bg["measured"]["particle_id"], bg["measured"]["accepted"]) if passed]
    variant_orders = {}
    for name, variant in variants.items():
        measured = variant["measured"]
        accepted_ids = [pid for pid, passed in zip(measured["particle_id"], measured["accepted"]) if passed]
        variant_orders[name] = {"source_only": shuffled(accepted_ids), "background_only": shuffled(background_ids),
                                "combined": shuffled(background_ids+accepted_ids)}
    warnings = ["Injection benchmark: canonical reservoir emission; no global energy or baryon conservation.",
                "Thermal background has no genuine flow or jets; a visible clump is not a critical-point measurement."]
    if request["background"]["provider"] == "hdf5":
        warnings.append("Native precision mixes stored background precision with float64 source. Packed preview cannot undo existing quantization.")
    data = {"schema_version": SCHEMA, "scenario_id": scenario_id(spec), "request_id": request["request_id"],
            "resolved_request": deepcopy(request), "provenance": {"core_version": core.CORE_VERSION,
            "rng_policy": core.RNG_POLICY, "numpy_version": np.__version__, "background_reference": request["background"],
            "thermal_law": core.THERMAL_LAW,
            "units": {"energy": "GeV", "momentum": "GeV", "p4_columns": ["E", "px", "py", "pz"]},
            "precision": "float64 core; float16 packed features; float32 packed event means"},
            "source_truth": [s.realized for s in samples], "background": bg, "variants": variants,
            "observation_order_by_variant": variant_orders,
            "observation_order": {"source_only": shuffled(source_ids), "background_only": shuffled(background_ids),
                                  "combined": shuffled(background_ids+source_ids)},
            "timing_ms": {"generation": (time.perf_counter()-start)*1000}, "warnings": warnings}
    if "event_mixed" in variants:
        data["provenance"]["event_mixing_policy"] = core.EVENT_MIXING_POLICY
    return ScenarioBundle(data)
