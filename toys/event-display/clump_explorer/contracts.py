"""Strict versioned request contracts, also usable by a future batch entry point."""
from copy import deepcopy
from dataclasses import dataclass
from .generator_adapter import core

SCHEMA = "hicad-clump-explorer-1"
BINNING = "proton_clump_diagnostics_v1"
COMPOSITIONS = ["source_only", "background_only", "combined"]
MODES = ["native", "packed_float16"]


def validate_background(b):
    core.fields(b, {"provider", "dataset_id", "event_index", "fingerprint"}, {"provider"}, "background")
    core.choice(b["provider"], ["none", "hdf5"], "background.provider")
    if b["provider"] == "none":
        core.fields(b, {"provider"}, {"provider"}, "background")
    else:
        core.fields(b, {"provider", "dataset_id", "event_index", "fingerprint"},
                    {"provider", "dataset_id", "event_index", "fingerprint"}, "background")
        for name in ("dataset_id", "fingerprint"):
            if not isinstance(b[name], str) or not 1 <= len(b[name]) <= 256:
                core.fail("background." + name, "Expected a nonempty public identifier")
        core.number(b["event_index"], "background.event_index", 0, 2**53-1, True)
    return deepcopy(b)


@dataclass(frozen=True)
class ScenarioSpec:
    request: dict

    @classmethod
    def parse(cls, value):
        if isinstance(value, cls):
            value = value.request
        keys = {"schema_version", "request_id", "realization", "background", "clump_config", "observation_mode", "diagnostics"}
        core.fields(value, keys | {"provenance"}, keys, "request")
        if "provenance" in value:
            p = value["provenance"]
            core.fields(p, {"core_version", "rng_policy", "numpy_version", "event_mixing_policy"}, {"core_version", "rng_policy"}, "provenance")
            core.choice(p["core_version"], [core.CORE_VERSION], "provenance.core_version")
            core.choice(p["rng_policy"], [core.RNG_POLICY], "provenance.rng_policy")
            if "numpy_version" in p:
                core.choice(p["numpy_version"], [core.np.__version__], "provenance.numpy_version")
            if "event_mixing_policy" in p:
                core.choice(p["event_mixing_policy"], [core.EVENT_MIXING_POLICY], "provenance.event_mixing_policy")
        core.choice(value["schema_version"], [SCHEMA], "schema_version")
        core.number(value["request_id"], "request_id", 0, 2**53-1, True)
        d = value["diagnostics"]
        core.fields(d, {"enabled", "pair_species", "binning_id"}, {"enabled", "pair_species", "binning_id"}, "diagnostics")
        if not isinstance(d["enabled"], bool):
            core.fail("diagnostics.enabled", "Expected boolean")
        core.choice(d["pair_species"], ["positive_proton"], "diagnostics.pair_species")
        core.choice(d["binning_id"], [BINNING], "diagnostics.binning_id")
        return cls({"schema_version": SCHEMA, "request_id": value["request_id"],
                    "realization": core.validate_realization(value["realization"]),
                    "background": validate_background(value["background"]),
                    "clump_config": core.validate_config(value["clump_config"]),
                    "observation_mode": core.choice(value["observation_mode"], MODES, "observation_mode"),
                    "diagnostics": deepcopy(d)})

    def physical(self):
        return {k: deepcopy(v) for k, v in self.request.items()
                if k not in ("request_id", "observation_mode", "diagnostics")}


def capabilities():
    return {"schema_version": SCHEMA, "source_schema": "hicad-clump-spec-1", "generator_version": core.CORE_VERSION,
            "thermal_law": core.THERMAL_LAW,
            "event_mixing_policy": core.EVENT_MIXING_POLICY,
            "rng_policy": core.RNG_POLICY, "numpy_version": core.np.__version__, "providers": ["none", "hdf5"], "variants": core.VARIANTS,
            "limits": {**core.LIMITS, "request_bytes": 65536, "cache_bundles": 32, "cache_bytes": 134217728,
                       "generation_concurrency": 2}, "precision_modes": MODES,
            "features": {"advanced_distributions": True, "deferred_exact_diagnostics": True, "event_mixed_control": True,
                         "hdf5_export": False, "detector_scoring": False}, "binning_id": BINNING}
