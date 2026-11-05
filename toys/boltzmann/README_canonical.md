# Backgrounds with the clump's massive thermal law

`generate_canonical_h5.py` creates independent thermal backgrounds using exactly
the same `clumps.sample_rest_particles` implementation as signal protons.
`sample_rest_protons` is its proton specialization; existing valid proton recipes
retain bit-for-bit samples after this refactor. The legacy `generate_thermal_h5.py`,
its README and its output datasets are preserved.

## Generate and inspect

From the checkout root:

```bash
./toys/boltzmann/regenerate_backgrounds.sh
python toys/event-display/serve_clump_events.py --data-dir toys/boltzmann/canonical-output --port 0 --open-browser
```

The shell script regenerates all three settings with 1,000 events each,
T = 100 MeV and seed 42, **replacing matching canonical output files**. It uses
`PYTHON` when supplied, otherwise an active virtual environment, the checkout's
`.venv`, or `python3`. It works from any working directory. All generator options
are forwarded, for example:

```bash
./toys/boltzmann/regenerate_backgrounds.sh --n-events 10000 --temperature-gev 0.15 --seed 123
./toys/boltzmann/regenerate_backgrounds.sh --settings bes --output-dir /tmp/canonical-bes
./toys/boltzmann/regenerate_backgrounds.sh --help
```

An explicit relative `--output-dir` is relative to your working directory. The
default remains `canonical-output` beside the generator. Use the Python entry
point directly without `--overwrite` when existing-file protection is desired.
Regenerated files have new fingerprints: reload the explorer and reselect the
background event before applying further signal changes.

Select a `canonical_thermal_events_*.h5` dataset. The explorer displays the recorded
**Background T** in MeV and distinguishes these files from legacy datasets with
unrecorded temperature metadata. Its default search includes both canonical and
legacy output directories; `--data-dir` limits the catalog to the chosen locations.

The generator's default output directory is `canonical-output` beside the script,
independent of the working directory. Output names prepend `canonical_` to the
legacy setting filename. Existing files are protected; use a separate
`--output-dir` for parameter studies. `--overwrite` explicitly permits replacement.
Files are assembled in a temporary path and published atomically. A failed run
does not leave a partially generated final dataset.

For a hotter sample without replacing the 100 MeV background:

```bash
python toys/boltzmann/generate_canonical_h5.py --settings bes --n-events 1000 --temperature-gev 0.15 --output-dir toys/boltzmann/canonical-T150
```

Available controls: `--settings`, `--n-events`, `--temperature-gev` (0.001–0.5),
decimal uint64 `--seed`, `--eta-max`, `--pt-min-gev`, optional `--pt-max-gev`,
`--compression lzf|gzip`, `--output-dir` and explicit `--overwrite`.

## Model

For each species of mass m, emission obeys

```text
dP(p) ∝ exp[-(sqrt(m² + |p|²) − m)/T] d³p
```

The shared exact kinetic-energy rejection sampler includes the p² radial
phase-space factor, isotropic directions and no truncated momentum tail. The
background uses pi±, K±, p and pbar with their actual masses. All species and all
events share the specified T; no old species-specific effective-temperature
weights or Gamma-pT law are applied. Eta is derived from the 3D momentum, then
lab acceptance is applied. There is no separate uniform-eta redraw.

Background particles are independent and at beta zero in the laboratory. A signal
clump uses the same law for protons, then applies its common Lorentz boost. Thus T
has the same model meaning in both sources. The final spectra can still differ
because boosts, masses or acceptance differ. At matched T and beta zero the
background proton and signal rest-frame distributions are the same; independent
RNG namespaces avoid creating identical particles when both seeds are 42.

The old energy-setting geometry, species abundances and multiplicity function are
reused as toy inputs. Here that function defines **emitted** multiplicity. Lab
cuts determine the stored **accepted** multiplicity; rejected particles are not
replaced to fill a target. Consequently, the old accepted-multiplicity and mean-pT
calibrations no longer hold. With beta zero, full azimuth and no pT cuts, an
isotropic source passes `abs(eta) < eta_max` with probability `tanh(eta_max)`
(about 0.762 at eta_max = 1). This is a controlled common-law benchmark, not a
new calibration to measured heavy-ion spectra. There is no genuine flow, jet
model, exact collision energy/baryon conservation or physical emission geometry.

## Data and reproducibility

The HDF5 schema retains:

- `mc_record`: float32 [2], collision energy and MC family label; never a signal flag.
- `event_record`: float32 [N,10], with actual accepted M and accepted-event mean/sum
  pT and mean eta. Legacy event-plane columns are placeholders (0,1), not real flow.
- `point_cloud`: float16 [N,capacity,5], normalized using the declared float32 event
  means. Active rows are [0:M] and all remaining rows are exactly zero.
- `observed/p4`: float32 [N,capacity,4], aligned `[E,px,py,pz]`, also zero padded.
- `truth/event/emitted_multiplicity`, `accepted_multiplicity` and float64
  `temperature_gev`, for generator bookkeeping separate from observed features.

Capacity grows without truncating events. Empty accepted events have M = 0,
zero summaries and padding; the explorer represents undefined averages as null.
The existing legacy validator/fit notebook has legacy-distribution assumptions;
use the new acceptance tests for these canonical datasets.

Root attributes store `thermal_law=massive_canonical_boltzmann_3d_v1`, generator and
core versions, input T and fixed-temperature policy, beta zero, acceptance, species,
probabilities, multiplicity/conservation policies, units, root seed, NumPy version
and RNG policy. Input temperature is explicit, not inferred from mean pT.

`background-seedsequence-pcg64-v1` uses SeedSequence entropy
`[root_low32, root_high32, 0x48494342, 1, setting_id, event_index, role, source_id,
daughter_id]`, then explicit PCG64. Setting IDs 1/2/3 identify LHC/RHIC/BES. Role 20
draws event geometry, role 10 species, roles 4/5 the shared radial/angular sampler.
Event streams are independent of event count, generation order and signal streams.
Replay requires the recorded versions. Increasing event count preserves existing
event samples. The pure `canonical_background.generate_background_event` function
can be reused without HDF5 or HTTP.

Run the physics/precision/reproducibility tests from the checkout root:

```bash
python -m unittest discover -s toys/event-display/clump_explorer/tests -v
```
