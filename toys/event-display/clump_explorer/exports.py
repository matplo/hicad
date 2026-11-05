"""Reproducible JSON exports; no HDF5 persistence during preview."""
from copy import deepcopy


def recipe(bundle):
    result = deepcopy(bundle.data["resolved_request"])
    result["provenance"] = {k: bundle.data["provenance"][k] for k in ("core_version", "rng_policy", "numpy_version")}
    if "event_mixing_policy" in bundle.data["provenance"]:
        result["provenance"]["event_mixing_policy"] = bundle.data["provenance"]["event_mixing_policy"]
    return result


def snapshot(bundle, spec=None):
    if spec is None:
        from .contracts import ScenarioSpec
        spec = ScenarioSpec.parse(bundle.data["resolved_request"])
    result = bundle.response(spec)
    result["export_type"] = "hicad-clump-snapshot-1"
    result["truth_note"] = "Source membership and rest p4 are simulation truth, separate from measured fields."
    return result


def model_observation(bundle, variant="shared_boost", composition="combined", mode="packed_float16"):
    """Future model adapter: no truth, origin-coded IDs or canonical precision leak."""
    observed = bundle.observation(variant, composition, mode)
    return {k: deepcopy(observed[k]) for k in ("p4", "species", "charge")}
