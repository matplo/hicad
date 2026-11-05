"""Exact fixed-bin positive-proton diagnostics with bounded pair workspaces."""
import numpy as np
from .generator_adapter import core
from .contracts import BINNING

DEFINITIONS = {"pt": (0, 3, 60, "GeV"), "y": (-2.5, 2.5, 50, "1"),
               "delta_phi": (-np.pi, np.pi, 72, "rad"), "abs_delta_y": (0, 5, 50, "1"),
               "q_inv": (0, 2, 80, "GeV"), "pair_beta_t": (0, 1, 50, "1")}


def q_inv(a, b):
    d = np.asarray(a)-np.asarray(b)
    q2 = np.sum(d[..., 1:]**2, axis=-1)-d[..., 0]**2
    tolerance = 1e-10*np.maximum(1, np.sum(d*d, axis=-1))
    if np.any(q2 < -tolerance):
        core.fail("diagnostics.q_inv", "Spacelike pair difference expected for equal-mass protons")
    return np.sqrt(np.maximum(0, q2))


class Histogram:
    def __init__(self, definition):
        lo, hi, n, self.units = definition
        self.edges = np.linspace(lo, hi, n+1)
        self.counts = np.zeros(n)
        self.underflow = self.overflow = self.total = 0.

    def add(self, values, weight=1.):
        values = np.asarray(values)
        if not np.isfinite(values).all():
            core.fail("diagnostics", "Nonfinite diagnostic values")
        self.counts += np.histogram(values, bins=self.edges)[0]*weight
        self.underflow += float(np.count_nonzero(values < self.edges[0]))*weight
        self.overflow += float(np.count_nonzero(values > self.edges[-1]))*weight
        self.total += values.size*weight

    def result(self):
        density = self.counts/(self.total*np.diff(self.edges)) if self.total else self.counts.copy()
        return {"counts": self.counts.tolist(), "edges": self.edges.tolist(), "underflow": self.underflow,
                "overflow": self.overflow, "total_entries": self.total, "density": density.tolist(),
                "units": self.units, "normalization": "density = counts / (total_entries including overflow * bin_width)",
                "edge_convention": "left inclusive; final right edge inclusive"}


def calculate(measured, clump_by_id=None, chunk=256):
    clump_by_id = clump_by_id or {}
    idx = [i for i, s in enumerate(measured["species"]) if s == "p"]
    p = np.asarray([measured["p4"][i] for i in idx], dtype=np.float64).reshape(-1, 4)
    groups = np.array([clump_by_id.get(measured["particle_id"][i], -1) for i in idx])
    k = core.kinematics(p)
    h = {name: Histogram(d) for name, d in DEFINITIONS.items()}
    truth = {name: Histogram(d) for name, d in DEFINITIONS.items() if name not in ("pt", "y")}
    h["pt"].add(k["pt"])
    h["y"].add(k["y"])
    pair_count = truth_count = 0
    for i in range(0, len(p), chunk):
        for j in range(i, len(p), chunk):
            aa = np.arange(i, min(i+chunk, len(p)))
            bb = np.arange(j, min(j+chunk, len(p)))
            ai, bi = np.meshgrid(aa, bb, indexing="ij")
            valid = ai < bi
            ai, bi = ai[valid], bi[valid]
            if not len(ai):
                continue
            a, b = p[ai], p[bi]
            delta_phi = (k["phi"][ai]-k["phi"][bi]+np.pi) % (2*np.pi)-np.pi
            values = {"delta_phi": delta_phi, "abs_delta_y": abs(k["y"][ai]-k["y"][bi]),
                      "q_inv": q_inv(a, b), "pair_beta_t": np.hypot(a[:, 1]+b[:, 1], a[:, 2]+b[:, 2])/(a[:, 0]+b[:, 0])}
            same = (groups[ai] >= 0) & (groups[ai] == groups[bi])
            pair_count += len(ai)
            truth_count += int(same.sum())
            for name, values_array in values.items():
                if name == "delta_phi":
                    for sign in (-1, 1):
                        wrapped = (sign*values_array+np.pi) % (2*np.pi)-np.pi
                        h[name].add(wrapped, .5)
                        truth[name].add(wrapped[same], .5)
                else:
                    h[name].add(values_array)
                    truth[name].add(values_array[same])
    return {"binning_id": BINNING, "pair_species": "positive_proton", "proton_count": len(p),
            "pair_count": pair_count, "truth_pair_count": truth_count, "exact": True,
            "message": "No proton pairs" if pair_count == 0 else f"{pair_count} unordered distinct proton pairs",
            "delta_phi_weights": {"positive": .5, "negative": .5},
            "histograms": {name: v.result() for name, v in h.items()},
            "truth_daughter_pairs": {"label": "Simulation truth: pairs from the same source clump; not an experimental observable",
                                     "histograms": {name: v.result() for name, v in truth.items()}}}


def for_bundle(bundle, variant, composition, mode):
    key = (mode, variant, composition, BINNING)
    with bundle.lock:
        if key in bundle.diagnostic_cache:
            return bundle.diagnostic_cache[key]
        source = bundle.data["variants"][variant]
        groups = dict(zip(source["measured"]["particle_id"], source["truth"]["clump_id"]))
        observation = bundle.observation(variant, composition, mode)
    # Pair work never holds the scene/cache lock: event replies and downloads
    # remain usable while an exact high-multiplicity histogram is running.
    result = calculate(observation, groups)
    if variant == "event_mixed":
        result["truth_daughter_pairs"].update(available=False,
            label="Independent donor events: no two mixed particles share a source; no same-clump pair overlay.")
        result["control_note"] = "Ensemble event-mixed reference; individual-event spectra and accepted counts may differ."
    with bundle.lock:
        bundle.diagnostic_cache[key] = result
    return result
