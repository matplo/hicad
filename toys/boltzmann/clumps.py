"""Massive canonical emission shared by proton clumps and thermal backgrounds.

Pure NumPy core, independent of HTTP and HDF5.

Four-vector columns are [E, px, py, pz], in GeV, with float64 arithmetic.
See RNG_POLICY and StreamFactory for the reproducibility contract.
"""
from dataclasses import dataclass
import math
import re

import numpy as np

PROTON_MASS = 0.938272088
THERMAL_LAW = "massive_canonical_boltzmann_3d_v1"
CORE_VERSION = "canonical-proton-1.0.0"
RNG_POLICY = "seedsequence-pcg64-roles-v1"
LIMITS = {"clumps": 8, "multiplicity_per_clump": 200, "temperature_min_gev": .001,
          "temperature_max_gev": .5, "beta_max": .95, "particles": 20000}
EVENT_MIXING_POLICY = "independent-source-donors-v1"
VARIANTS = ["shared_boost", "azimuth_decohered", "event_mixed"]


class ParameterError(ValueError):
    def __init__(self, field, message, code="invalid_parameter", status=422):
        super().__init__(message)
        self.field, self.code, self.status = field, code, status


def fail(field, message):
    raise ParameterError(field, message)


def fields(value, allowed, required, path):
    if not isinstance(value, dict):
        fail(path, "Expected an object")
    for name in value:
        if name not in allowed:
            fail(f"{path}.{name}", "Unknown field")
    for name in required:
        if name not in value:
            fail(f"{path}.{name}", "Required field")


def number(value, path, lo=None, hi=None, integer=False):
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        fail(path, "Expected an integer" if integer else "Expected a finite number")
    try:
        finite = math.isfinite(value)
    except OverflowError:
        finite = False
    if not finite or (integer and not isinstance(value, int)):
        fail(path, "Expected a finite integer" if integer else "Expected a finite number")
    if (lo is not None and value < lo) or (hi is not None and value > hi):
        fail(path, f"Must be within [{lo}, {hi}]")
    return value


def choice(value, choices, path):
    if not isinstance(value, str) or value not in choices:
        fail(path, "Supported values: " + ", ".join(choices))
    return value


def seed(value, path="realization.root_seed"):
    if not isinstance(value, str) or not re.fullmatch(r"[0-9]{1,20}", value) or int(value) >= 2**64:
        fail(path, "Expected a decimal uint64 string")
    return str(int(value))


def validate_realization(r):
    keys = {"root_seed", "clump_revision", "direction_revision", "control_revision"}
    fields(r, keys, keys, "realization")
    out = {"root_seed": seed(r["root_seed"])}
    for k in keys - {"root_seed"}:
        out[k] = number(r[k], "realization." + k, 0, 2**32-1, True)
    return out


class StreamFactory:
    """Independent stable roles: n=1, T=2, speed=3, radial=4, angle=5,
    collective=6, control=7, permutation=8. Direction ignores clump_revision.
    Each SeedSequence entropy is [root low32, root high32, role, revision,
    clump ID, daughter ID]. PCG64 is explicit, never NumPy's implicit default.
    """
    def __init__(self, realization):
        self.r = validate_realization(realization)

    def rng(self, role, clump_id=0, daughter_id=0):
        r = self.r
        revision = r["direction_revision"] if role == 6 else r["clump_revision"]
        root = int(r["root_seed"])
        entropy = [root & 0xffffffff, root >> 32, role, revision, clump_id, daughter_id]
        if role in (7, 8):
            entropy.append(r["control_revision"])
        return np.random.Generator(np.random.PCG64(np.random.SeedSequence(entropy)))


class EventMixedStreams:
    """Disjoint synthetic donor events, one donor per target daughter slot.

    Roles retain their usual meanings. Control revision redraws donors; direction
    revision only changes donor directions. No donor shares parameter draws with
    another, and no RNG is coupled to the background or target source sample.
    """
    def __init__(self, realization, slot_clump_id, slot_daughter_id):
        self.r = validate_realization(realization)
        self.slot_clump_id, self.slot_daughter_id = slot_clump_id, slot_daughter_id

    def rng(self, role, clump_id=0, daughter_id=0):
        r = self.r
        root = int(r["root_seed"])
        revision = r["direction_revision"] if role == 6 else r["clump_revision"]
        entropy = [root & 0xffffffff, root >> 32, 0x4d495831, 1, role, revision,
                   self.slot_clump_id, self.slot_daughter_id, clump_id, daughter_id,
                   r["control_revision"]]
        return np.random.Generator(np.random.PCG64(np.random.SeedSequence(entropy)))


def validate_distribution(d, path, lo, hi, integer=False):
    fields(d, {"distribution", "value", "minimum", "maximum", "offset", "mean", "truncation"},
           {"distribution"}, path)
    modes = ["fixed", "uniform_integer", "shifted_poisson_truncated"] if integer else ["fixed", "uniform"]
    mode = choice(d["distribution"], modes, path + ".distribution")
    keys = {"fixed": {"distribution", "value"},
            "uniform": {"distribution", "minimum", "maximum"},
            "uniform_integer": {"distribution", "minimum", "maximum"},
            "shifted_poisson_truncated": {"distribution", "offset", "mean", "maximum", "truncation"}}[mode]
    required = keys - ({"truncation"} if mode == "shifted_poisson_truncated" else set())
    fields(d, keys, required, path)
    out = dict(d)
    if mode == "fixed":
        number(d["value"], path + ".value", lo, hi, integer)
    elif mode in ("uniform", "uniform_integer"):
        a = number(d["minimum"], path + ".minimum", lo, hi, integer)
        b = number(d["maximum"], path + ".maximum", lo, hi, integer)
        if b < a or (not integer and b == a):
            fail(path, "Invalid distribution interval")
    else:
        offset = number(d["offset"], path + ".offset", lo, hi, True)
        maximum = number(d["maximum"], path + ".maximum", offset, hi, True)
        mean = number(d["mean"], path + ".mean", 0, 10000)
        choice(d.get("truncation", "reject_and_redraw"), ["reject_and_redraw"], path + ".truncation")
        if mean:
            logs = [-mean + k * math.log(mean) - math.lgamma(k+1) for k in range(maximum-offset+1)]
            top = max(logs)
            log_p = top + math.log(sum(math.exp(x-top) for x in logs))
            if log_p < math.log(.0001):
                fail(path, "Truncation probability below 0.0001; rejection cost exceeds service limit")
        out["truncation"] = "reject_and_redraw"
    return out


def draw_distribution(d, rng):
    mode = d["distribution"]
    if mode == "fixed":
        return d["value"]
    if mode == "uniform":
        return float(rng.uniform(d["minimum"], d["maximum"]))
    if mode == "uniform_integer":
        return int(rng.integers(d["minimum"], d["maximum"]+1))
    for _ in range(100000):
        n = int(d["offset"] + rng.poisson(d["mean"]))
        if n <= d["maximum"]:
            return n
    fail("emitted_multiplicity", "Truncated Poisson rejection budget exhausted")


def validate_acceptance(a):
    path = "clump_config.acceptance"
    keys = {"eta_max", "pt_min_gev", "pt_max_gev", "azimuth"}
    fields(a, keys, keys, path)
    number(a["eta_max"], path + ".eta_max", 1e-12, 100)
    number(a["pt_min_gev"], path + ".pt_min_gev", 0, 1e6)
    if a["pt_max_gev"] is not None:
        number(a["pt_max_gev"], path + ".pt_max_gev", 0, 1e6)
        if a["pt_max_gev"] <= a["pt_min_gev"]:
            fail(path + ".pt_max_gev", "Must exceed pT minimum")
    choice(a["azimuth"], ["full"], path + ".azimuth")
    return dict(a)


def validate_config(c):
    path = "clump_config"
    keys = {"schema_version", "enabled", "signal_probability", "injection_mode", "source_frame_mode",
            "species", "clumps_per_injected_event", "emitted_multiplicity", "temperature_gev", "boost_speed",
            "boost_direction", "acceptance", "variants", "shuffle_particles", "write_truth",
            "write_observed_p4_float32", "overwrite"}
    fields(c, keys, keys, path)
    out = dict(c)
    for name in ["enabled", "shuffle_particles", "write_truth", "write_observed_p4_float32", "overwrite"]:
        if not isinstance(c[name], bool):
            fail(path + "." + name, "Expected boolean")
    if not c["shuffle_particles"]:
        fail(path + ".shuffle_particles", "Live observations require aligned shuffling to avoid encoding source membership in row blocks")
    for name, values in {"schema_version": ["hicad-clump-spec-1"], "injection_mode": ["add"],
                         "source_frame_mode": ["canonical_source"], "species": ["p"],
                         "boost_direction": ["transverse", "isotropic_3d"]}.items():
        choice(c[name], values, path + "." + name)
    number(c["signal_probability"], path + ".signal_probability", 0, 1)
    # Non-unit probability is a batch policy, deliberately unsupported by this preview core.
    if c["signal_probability"] != 1:
        fail(path + ".signal_probability", "Live preview requires 1; use enabled for deterministic on/off")
    for name, lo, hi, integer in [("clumps_per_injected_event", 0, 8, True),
                                   ("emitted_multiplicity", 0, 200, True),
                                   ("temperature_gev", .001, .5, False), ("boost_speed", 0, .95, False)]:
        out[name] = validate_distribution(c[name], path + "." + name, lo, hi, integer)
    if c["clumps_per_injected_event"]["distribution"] != "fixed":
        fail(path + ".clumps_per_injected_event", "Only fixed clump count supported in v1")
    out["acceptance"] = validate_acceptance(c["acceptance"])
    if not isinstance(c["variants"], list) or not c["variants"]:
        fail(path + ".variants", "Expected a nonempty array")
    for v in c["variants"]:
        choice(v, VARIANTS, path + ".variants")
    if len(set(c["variants"])) != len(c["variants"]):
        fail(path + ".variants", "Duplicate variant")
    out["variants"] = [v for v in VARIANTS if v in c["variants"]]
    return out


def sample_rest_protons(n, temperature_gev, stream_factory, clump_id=0):
    """Proton specialization of the shared all-species canonical sampler.

    This wrapper retains the original proton stream mapping bit for bit.
    """
    return sample_rest_particles(n, PROTON_MASS, temperature_gev, stream_factory, clump_id)


def sample_rest_particles(n, mass_gev, temperature_gev, stream_factory, source_id=0):
    """Sample exp[-(E-m)/T] d³p for a scalar mass or one mass per daughter.

    Backgrounds and clumps use this same rejection sampler; their independent
    stream factories and collective boosts define different source ensembles.
    """
    number(n, "n", 0, 1000000, True)
    t = number(temperature_gev, "temperature_gev", 1e-12, 1e6)
    if isinstance(mass_gev, bool):
        fail("mass_gev", "Expected positive finite masses")
    try:
        if np.asarray(mass_gev).dtype.kind not in "iuf":
            fail("mass_gev", "Expected numeric masses, not booleans or strings")
        masses = np.asarray(mass_gev, dtype=np.float64)
    except (TypeError, ValueError):
        fail("mass_gev", "Expected a scalar mass or one mass per particle")
    if masses.ndim not in (0, 1) or (masses.ndim == 1 and masses.shape != (n,)):
        fail("mass_gev", "Expected a scalar mass or one mass per particle")
    if not np.isfinite(masses).all() or np.any(masses <= 0) or np.any(masses > 1e6):
        fail("mass_gev", "Masses must be positive, finite and no greater than 1e6 GeV")
    if masses.ndim == 0:
        masses = np.full(n, float(masses))
    result = np.empty((n, 4), dtype=np.float64)
    for i in range(n):
        m = float(masses[i])
        weights = np.array([m*m*t, 2*m*t*t, 2*t**3])
        weights /= weights.sum()
        radial = stream_factory.rng(4, source_id, i)
        for _ in range(100000):
            k = radial.gamma(int(radial.choice([1, 2, 3], p=weights)), t)
            p = math.sqrt(k * (k + 2*m))
            if radial.random() < p / (k+m):
                break
        else:
            fail("temperature_gev", "Radial rejection budget exhausted")
        angular = stream_factory.rng(5, source_id, i)
        z = angular.uniform(-1, 1)
        phi = angular.uniform(-np.pi, np.pi)
        transverse = p * math.sqrt(max(0, 1-z*z))
        result[i] = [k+m, transverse*math.cos(phi), transverse*math.sin(phi), p*z]
    return result


def sample_boost_vector(speed, direction_mode, rng):
    number(speed, "speed", 0, np.nextafter(1., 0.))
    choice(direction_mode, ["transverse", "isotropic_3d"], "direction_mode")
    phi = rng.uniform(-np.pi, np.pi)
    z = rng.uniform(-1, 1) if direction_mode == "isotropic_3d" else 0.
    return speed * np.array([math.sqrt(1-z*z)*math.cos(phi), math.sqrt(1-z*z)*math.sin(phi), z])


def boost_four_vectors(p4, beta):
    p4, beta = np.asarray(p4, dtype=np.float64), np.asarray(beta, dtype=np.float64)
    if p4.ndim != 2 or p4.shape[1] != 4 or not np.isfinite(p4).all():
        fail("p4", "Expected finite [N,4] four-vectors")
    if beta.shape != (3,) or not np.isfinite(beta).all() or beta @ beta >= 1:
        fail("beta", "Expected finite subluminal three-vector")
    gamma = 1 / math.sqrt(1 - beta @ beta)
    dot = p4[:, 1:] @ beta
    return np.column_stack((gamma*(p4[:, 0]+dot),
                            p4[:, 1:] + (gamma*p4[:, 0] + gamma**2/(gamma+1)*dot)[:, None]*beta))


def kinematics(p4):
    p4 = np.asarray(p4, dtype=np.float64).reshape(-1, 4)
    pt = np.hypot(p4[:, 1], p4[:, 2])
    mass2 = p4[:, 0]**2 - np.sum(p4[:, 1:]**2, axis=1)
    mt = np.sqrt(np.maximum(0., mass2) + pt*pt)
    eta = np.full(len(pt), np.nan)
    phi = np.full(len(pt), np.nan)
    y = np.full(len(pt), np.nan)
    np.divide(p4[:, 3], pt, out=eta, where=pt > 0)
    eta = np.arcsinh(eta)
    phi[pt > 0] = np.arctan2(p4[pt > 0, 2], p4[pt > 0, 1])
    np.divide(p4[:, 3], mt, out=y, where=mt > 0)
    return {"pt": pt, "eta": eta, "phi": phi, "y": np.arcsinh(y)}


def apply_acceptance(p4, acceptance):
    a = validate_acceptance(acceptance)
    k = kinematics(p4)
    mask = np.isfinite(k["eta"]) & (np.abs(k["eta"]) < a["eta_max"]) & (k["pt"] >= a["pt_min_gev"])
    if a["pt_max_gev"] is not None:
        mask &= k["pt"] < a["pt_max_gev"]
    return mask


@dataclass
class ClumpSample:
    rest_p4: np.ndarray
    lab_p4: np.ndarray
    accepted: np.ndarray
    particle_id: list
    realized: dict


def generate_clump(config, realization, clump_id=0):
    config = validate_config(config)
    streams = StreamFactory(realization)
    number(clump_id, "clump_id", 0, 7, True)
    n = draw_distribution(config["emitted_multiplicity"], streams.rng(1, clump_id)) if config["enabled"] else 0
    t = draw_distribution(config["temperature_gev"], streams.rng(2, clump_id))
    speed = draw_distribution(config["boost_speed"], streams.rng(3, clump_id))
    rest = sample_rest_protons(n, t, streams, clump_id)
    beta = sample_boost_vector(speed, config["boost_direction"], streams.rng(6, clump_id))
    lab = boost_four_vectors(rest, beta)
    accepted = apply_acceptance(lab, config["acceptance"])
    realized = {"clump_id": clump_id, "emitted": n, "accepted": int(accepted.sum()),
                "temperature_gev": t, "boost_speed": speed, "beta_vector": beta.tolist(),
                "undefined_eta_rejections": int(np.count_nonzero(~np.isfinite(kinematics(lab)["eta"])))}
    return ClumpSample(rest, lab, accepted, [f"c{clump_id}:d{i}" for i in range(n)], realized)


@dataclass
class EventMixedSample(ClumpSample):
    donors: dict


def generate_event_mixed_clump(config, realization, clump_id=0):
    """Synthetic event mixing: each slot draws from an independent source.

    Match the target emitted n, not its accepted count. For each donor draw T,
    speed and direction independently from the same configured distributions,
    sample one rest proton, boost its full p4, then apply the same lab cuts.
    Current n/T/speed distributions are independent, so this reproduces the
    particle-weighted one-body ensemble while removing shared source parameters.
    It does not preserve a particular event's pT/y arrays or same-source pairs.
    """
    config = validate_config(config)
    number(clump_id, "clump_id", 0, 7, True)
    target_streams = StreamFactory(realization)
    n = draw_distribution(config["emitted_multiplicity"], target_streams.rng(1, clump_id)) if config["enabled"] else 0
    rest, lab = np.empty((n, 4)), np.empty((n, 4))
    temperatures, speeds, betas, donor_ids = [], [], [], []
    for daughter in range(n):
        streams = EventMixedStreams(realization, clump_id, daughter)
        t = draw_distribution(config["temperature_gev"], streams.rng(2))
        speed = draw_distribution(config["boost_speed"], streams.rng(3))
        beta = sample_boost_vector(speed, config["boost_direction"], streams.rng(6))
        particle = sample_rest_protons(1, t, streams)
        rest[daughter], lab[daughter] = particle[0], boost_four_vectors(particle, beta)[0]
        temperatures.append(t); speeds.append(speed); betas.append(beta.tolist())
        donor_ids.append(f"donor:c{clump_id}:d{daughter}")
    accepted = apply_acceptance(lab, config["acceptance"])
    realized = {"clump_id": clump_id, "emitted": n, "accepted": int(accepted.sum()), "independent_donors": n,
                "temperature_gev": config["temperature_gev"].get("value"),
                "boost_speed": config["boost_speed"].get("value"),
                "temperature_range_gev": [min(temperatures), max(temperatures)] if n else None,
                "boost_speed_range": [min(speeds), max(speeds)] if n else None}
    donors = {"donor_event_id": donor_ids, "temperature_gev": temperatures, "boost_speed": speeds,
              "beta_vectors": betas, "slot_clump_id": [clump_id]*n}
    return EventMixedSample(rest, lab, accepted, [f"m:c{clump_id}:d{i}" for i in range(n)], realized, donors)
