"""Independent background emission using the exact clump thermal sampler.

No radial-flow approximation or species-specific slope temperature. All species
share a specified thermodynamic model parameter T; masses enter the 3D law.
Geometry/multiplicity are optional orchestration concerns in the HDF5 entry point.
"""
from dataclasses import dataclass
import numpy as np
import clumps

VERSION = "canonical-background-1.0.0"
RNG_POLICY = "background-seedsequence-pcg64-v1"
SPECIES = (("pi+", .13957039, 1), ("pi-", .13957039, -1),
           ("K+", .493677, 1), ("K-", .493677, -1),
           ("p", clumps.PROTON_MASS, 1), ("pbar", clumps.PROTON_MASS, -1))
MASS = np.array([s[1] for s in SPECIES])
CHARGE = np.array([s[2] for s in SPECIES])


class BackgroundStreams:
    """Disjoint namespace from clump RNGs; stable across event/order/count changes."""
    def __init__(self, root_seed, event_index=0, setting_id=0):
        self.root = int(clumps.seed(root_seed, "root_seed"))
        self.event_index = clumps.number(event_index, "event_index", 0, 2**32-1, True)
        self.setting_id = clumps.number(setting_id, "setting_id", 0, 2**32-1, True)

    def rng(self, role, source_id=0, daughter_id=0):
        entropy = [self.root & 0xffffffff, self.root >> 32, 0x48494342, 1,
                   self.setting_id, self.event_index, role, source_id, daughter_id]
        return np.random.Generator(np.random.PCG64(np.random.SeedSequence(entropy)))


@dataclass
class BackgroundSample:
    emitted_p4: np.ndarray
    accepted: np.ndarray
    species_index: np.ndarray

    @property
    def accepted_p4(self):
        return self.emitted_p4[self.accepted]


def generate_background_event(n_emitted, temperature_gev, species_probabilities,
                              root_seed="42", event_index=0, setting_id=0,
                              acceptance=None, stream_factory=None):
    """Independent massive canonical particles at rest in the lab, then lab cuts.

    n_emitted is before acceptance, never a target to refill by rejection. The
    optional stream_factory supports exact sampler-parity tests and batch reuse.
    """
    clumps.number(n_emitted, "n_emitted", 0, 20000, True)
    clumps.number(temperature_gev, "temperature_gev", .001, .5)
    probabilities = np.asarray(species_probabilities, dtype=np.float64)
    if probabilities.shape != (6,) or not np.isfinite(probabilities).all() or np.any(probabilities < 0) or probabilities.sum() <= 0:
        clumps.fail("species_probabilities", "Expected six finite nonnegative probabilities with a positive sum")
    probabilities = probabilities / probabilities.sum()
    streams = stream_factory if stream_factory is not None else BackgroundStreams(root_seed, event_index, setting_id)
    indices = streams.rng(10).choice(6, size=n_emitted, p=probabilities)
    p4 = clumps.sample_rest_particles(n_emitted, MASS[indices], temperature_gev, streams)
    acceptance = acceptance if acceptance is not None else {"eta_max": 1., "pt_min_gev": 0., "pt_max_gev": None, "azimuth": "full"}
    return BackgroundSample(p4, clumps.apply_acceptance(p4, acceptance), indices)
