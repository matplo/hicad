# Canonical proton-clump explorer

The new explorer adds a server-side canonical proton source, pinned HDF5 backgrounds,
shared/decohered/event-mixed comparisons, exact positive-proton diagnostics and reproducible JSON
downloads. The original server, HTML, documentation and thermal generator are untouched.

## Run

From the checkout root, with Python 3.10 or newer:

```bash
python -m pip install -r requirements.txt
python toys/event-display/serve_clump_events.py --port 8766 --open-browser
```

The default URL is `http://127.0.0.1:8766/`. The existing display can continue on
8765. The server resolves assets and the shared core relative to its own files,
so it can also be launched with an absolute script path from another directory.
`henv -x` or `.venv/bin/python` can be used in place of `python`.

If 8766 is occupied, choose another port or let the operating system assign a
free one. The server prints and opens the actual selected URL:

```bash
python toys/event-display/serve_clump_events.py --port 0 --open-browser
```

The default catalog searches `toys/boltzmann/canonical-output` and the legacy
`toys/boltzmann/output`, relative to the checkout.
To use other allowed datasets, repeat `--data-dir` with directories or `.h5` files:

```bash
python toys/event-display/serve_clump_events.py --data-dir toys/boltzmann/canonical-output --port 8766
```

Source-only operation needs no HDF5 files and does not import h5py. NumPy is the
only dependency of the source core. Existing project plotting scripts and the
independent radial-law tests additionally use SciPy/Matplotlib from the root
requirements. The browser uses the legacy display's Three.js **0.171.0** CDN
version for optional WebGPU 3D. Controls, Canvas plots, diagnostics and downloads
initialize independently and continue when the GPU or CDN is unavailable.

New backgrounds use the same massive canonical 3D sampler as the signal, through
`generate_canonical_h5.py`; the explorer generates the signal clumps live. A small
example, from the checkout root:

```bash
python toys/boltzmann/generate_canonical_h5.py --settings bes --n-events 1000 --temperature-gev 0.1
python toys/event-display/serve_clump_events.py --data-dir toys/boltzmann/canonical-output --port 0 --open-browser
```

Without `--data-dir`, the two directories above are searched independently of the
launch directory. Every supplied directory is scanned for `*.h5` and
`output/*.h5`; individual files are also accepted. Reload the page to refresh the
catalog after generating more data. The legacy `generate_thermal_h5.py` and its
existing Gamma-pT datasets remain unchanged and do not become canonical by loading
them. The dataset selector and background readout show recorded temperatures;
legacy files without those attributes show “T unrecorded”.

The canonical background has fixed T = 100 MeV by default, common to all species
and events, with beta zero and independent particle emission. Signal clumps use
the same rest-frame law and their configurable common boost. The **Background T**
metric shows the recorded model input; it is not a slope estimate or half mean pT.
Equal T with different boosts does not imply equal lab spectra. See
[canonical background details](../boltzmann/README_canonical.md) for generation,
RNG and multiplicity semantics. Choose a different output directory to create a
different background temperature without replacing an existing dataset.

## Controls and interpretation

- Start with eight emitted protons, T = 100 MeV and a common transverse β = 0.4.
  Sliders have numeric entries. Temperature is converted to GeV for the API;
  measured mean pT is an output, not 2T for this massive source.
- **Help & physics** opens a modal; the **?** beside beta opens the same help.
  Beta is the clump source speed v/c: at 0.4 it moves at 40% of light speed.
  Every daughter gets the same Lorentz boost in the sampled direction, while
  retaining its individual thermal momentum. Beta zero removes collective motion
  while retaining thermal motion. The modal explains this, the controls,
  comparison modes, colors, backgrounds, replay and recipe downloads. Escape,
  either Close button or a click outside the modal closes it.
- **New clump** redraws rest momenta and variable n/T/speed, preserving the
  collective direction. **New direction** changes only its stream. A direction
  distribution change uses the same direction seeds with the new mapping.
- **New control draws** increments control revision, redrawing azimuth scrambling
  and independent donor events while preserving the shared clump and background.
- **Include phi-scrambled control** and **Include event-mixed control** are
  separate opt-in checkboxes, both unchecked initially. The default recipe only
  requests `shared_boost`, so unused controls are not generated. Loading a saved
  recipe restores its variant selection; the checkboxes and Advanced JSON stay
  synchronized. Toggling either control preserves the base source and background.
- **Compare → Shared / event mixed** shows the original source and independent
  donor reference together after enabling the mixed checkbox; **Enabled views**
  shows whichever samples are requested, including azimuth scrambling when enabled.
  Each panel reports its own emitted, accepted and displayed counts. Header
  metrics explicitly name the first displayed variant. Old recipes retain their
  variant list; tick the mixed checkbox to add `event_mixed`.
- Lab cuts are strict `abs(eta) < eta_max`, inclusive pT minimum and optional
  exclusive pT maximum. They select the existing emitted sample. Rejected
  daughters are available as explicitly labelled truth ghosts.
  Beta changes lab pT and eta even when rest p4 is fixed, so a daughter may leave
  and later re-enter acceptance at a different laboratory momentum. For seed 42,
  the default daughter `c0:d4` is accepted at beta 0.17, rejected at 0.18 through
  0.68, and accepted again at 0.69. Its rest p4 and pz stay identical throughout;
  px passes from positive to negative while hidden. Enable **Rejected daughters**
  to follow its gray ghost. This is selection, not a new random daughter.
- Advanced JSON edits the entire nested `clump_config`. **Variable example**
  loads fixed-count clumps with truncated shifted-Poisson n and uniform T/β.
  Changing a basic n/T/β input replaces that field with a fixed distribution.
  A saved recipe can be loaded using the file input and posted directly to the API.
- Select a background dataset, centrality window and event, then pin it using
  navigation or **Pin index**. Random selection uses the decimal root seed as
  its independent selection seed; repeated identical random-selection requests
  deliberately resolve to the same event. Change the seed for another random
  event, or use next/previous. Changing n/T/β does not reselect the background.
- Composition, variant comparison, frame, coloring, fixed axes, ghosts and camera
  are local view controls. Background visibility does not change source p4 or
  camera. **Fit compared union** is an explicit fit over all requested variants and
  components. Cameras are linked; scenario updates preserve camera/axis choices.
- 3D coordinates are `(px,py,pz)` in GeV; the optional pz multiplier is shown in
  the readout. Canvas views show px–py and rapidity y versus wrapped φ−φc, where
  φc is the first source's common boost azimuth. Centering the circular coordinate
  avoids splitting a clump crossing the original −π/π boundary. All compared
  views share scales. Points beyond fixed Canvas axes are counted on the plot.
  The 3D box is only a guide and does not clip the event. Each panel separately
  reports emitted source particles rejected by lab cuts and accepted source
  **centers** outside the camera's viewport/near/far planes. Viewport counts do
  not account for depth occlusion by overlapping particles. Camera framing stays
  fixed on beta changes. **Fit compared union** explicitly fits a bounding sphere
  containing the complete coordinate box/data using the limiting horizontal or
  vertical FOV (including narrow screens); linked views use a common distance.
  The far plane grows when fitting or zooming out requires it, without moving
  points, altering acceptance or automatically moving the camera.
- Default **Clump / background · truth** coloring shows clump protons in red and
  background protons in green. This reveals simulation truth, as does the source
  direction arrow. **Species · origin hidden** colors protons alike for a neutral
  comparison; pbar remain distinct. Rest-frame views contain source
  daughters only, with their *lab* acceptance mask; no rest acceptance is applied.
  Main observed summaries and pair diagnostics always refer to the selected lab
  observation, even during this separate rest-frame truth inspection.
  In an event-mixed panel, red points are independent source-donor protons; there
  is no collective direction arrow or single rest frame. Its rest inspection
  displays each proton in its individual donor frame and labels this explicitly.
- **Replay illustration** reveals momentum vectors, without a time or position
  interpretation. **Replay on change** is unchecked initially; tick it to replay
  after a changed event is committed. Local color, camera and composition changes
  do not trigger replay or generation. The manual Replay button always remains
  available. Canvas transverse points reveal from the origin and y–phi points
  fade in, so replay also works without WebGPU. Authoritative p4 and statistics
  remain fixed throughout the illustration.
- **Auto update** debounces physical changes by 200 ms, with immediate trailing
  submission on release or Apply. There is one in-flight request and one latest
  pending draft. Draft values update immediately; plots/statistics stay on the
  last complete committed event. Reset restores the initial physical recipe,
  invalidates pending replies, and preserves local camera/view choices.
- **Save recipe…** beside Apply (also **Save recipe JSON** under Downloads)
  downloads the displayed event's physical settings, exact seeds, versions and
  pinned background reference as a loadable JSON file. Saving uses the committed
  browser copy directly, even after server cache eviction. It never captures an
  uncommitted draft. Snapshot downloads additionally contain particle p4 and truth.
  The PNG exports labelled Canvas comparisons, including without WebGPU.

The operational limits are 8 clumps, 200 emitted daughters per clump,
0.001–0.5 GeV temperature, β ≤ 0.95 and 20,000 supplied particles including
background and rejected daughters. All particles are rendered; no physical or
render-only subsampling is performed. Requests exceeding limits fail explicitly.
Live preview requires `signal_probability=1` and aligned `shuffle_particles=true`.
Use `enabled=false` for deterministic source off. Persistence flags are metadata
and do not write datasets. Unsupported replacement, exact decay, nonuniform
azimuth acceptance, the separate unsupported `independent_boost` mode or detector
modes are rejected. `event_mixed` is the explicitly defined independent-source
reference below, rather than a silently substituted boost mode.

## Physics and random streams

The shared core is `toys/boltzmann/clumps.py`, with four-vectors `[E,px,py,pz]`
in GeV and internal float64 arithmetic. It implements the specified massive
canonical Boltzmann law `exp(-(E−m)/T) d³p`, using the exact Gamma-mixture kinetic
energy rejection sampler with its p² phase-space factor and an untruncated tail.
Rest emission is isotropic. Every daughter's lab four-vector uses the same
Lorentz boost; cuts are applied afterward. The angular twin independently
scrambles each daughter's azimuth while preserving pT, eta, energy, ID and lab
acceptance. Scrambling is indexed by stable daughter identity, including rejected
daughters, so acceptance changes cannot redraw remaining control angles.

“Shared” means the boost speed and direction are common to daughters of each
clump. “Azimuth decohered” means independent lab-frame φ randomization of those
same daughters around the beam axis, preserving pT, eta, energy and acceptance.
The background is identical in both views. It removes the source's azimuthal
alignment and can change pair opening angles/q_inv. It does not model quantum
decoherence. Several clumps may each have their own shared boost.

### Event-mixed control

`event_mixed` implements **synthetic independent-source event mixing**. Match the
target emitted n in each source slot. For every slot, draw an independent donor
source's T, speed and direction from the configured distributions, sample one
massive thermal proton, apply its full Lorentz boost, and then the same lab cuts.
Donors use a separate RNG namespace. No saved background pool, disk dataset or
previous HTTP request is used. The background reference remains identical.

This removes correlations due to a common boost direction (including longitudinal
motion for isotropic 3D directions) and fluctuating shared T/speed. The current
multiplicity, T and speed distributions are independent, so independent donors
match the source's particle-weighted one-body distribution **over the ensemble**.
Individual events need not have identical pT/y histograms, accepted counts or
total four-momentum. The control is not a kinematic twin. It is also not a uniform
rapidity redraw or a within-clump permutation, which would leave all pairwise
|Δy| values unchanged. For fixed T/speed and transverse boosts there is no shared
longitudinal boost; a large longitudinal difference is not expected.

Mixed IDs `m:cK:dJ` identify distinct donor particles, not the physical daughters
in the shared/azimuth pair. Truth records unique `donor_event_id`, per-donor T,
speed, beta vectors and rest p4; `slot_clump_id` only identifies the target count
slot. Physical `clump_id` is -1, so artificial same-source pairs cannot enter the
red truth histogram. The mixed panel has no red same-clump overlay. Background
IDs and arrays are common, and observations have stable variant-specific ordering.

Policy `independent-source-donors-v1` uses SeedSequence entropy
`[root_low32, root_high32, 0x4d495831, 1, role, revision, slot_clump_id,
slot_daughter_id, local_clump_id, local_daughter_id, control_revision]`, followed
by explicit PCG64. Roles 2/3 draw donor T/speed, 4/5 rest radial/angular momenta,
6 direction, 8 permutation ranks. Revision is direction revision for role 6 and
clump revision otherwise. Control revision redraws all donor streams. Increasing
fixed n preserves previous donors; beta, direction and acceptance changes retain
the required rest-sample coupling. Recipes/capabilities record the mixing policy;
its version also enters IDs of scenarios requesting this variant. Older two-view
recipes and their physics remain supported.

Root seeds are decimal uint64 **strings**, including `18446744073709551615`.
Revision keys are nonnegative uint32 integers. The versioned policy
`seedsequence-pcg64-roles-v1` uses explicit NumPy PCG64 generators and SeedSequence
entropy `[root_low32, root_high32, role, revision, clump_id, daughter_id]`.
Roles 1/2/3 draw n/T/speed; 4/5 draw per-daughter radial/angular momenta; 6 draws
the collective direction; 7 scrambles azimuth; 8 orders observations. Role 6 uses
direction revision, while other roles use clump revision. Roles 7/8 additionally
include control revision. Background permutation uses the reserved clump key 8;
physical source clump keys are 0–7. No mutable RNG is shared between requests.

Changing beta preserves rest momenta and collective direction. Changing direction
preserves rest momenta and speed. Increasing fixed n or clump count preserves
existing daughters/clumps. Changing temperature reuses the same seeds but may
change rejection histories; resulting dots are not continuous physical paths.
Background selection uses a separate explicitly seeded PCG64 selection operation
and resolves to an exact pinned reference before generation.

Replay requires the saved generator/RNG versions and NumPy version; exported
recipes record these and reject incompatible versions. Scenario IDs hash canonical
physical configuration, realization, pinned background, generator/RNG and NumPy
versions. Request IDs, observation precision, diagnostics and local view state do
not change the physical ID. Two processes using the same versions and inputs
produce the same arrays.

For future batch use, import `ScenarioSpec` and `generate_scenario` from
`clump_explorer` with `toys/event-display` on the Python import path. The API calls
this same transport-independent function. `exports.model_observation` removes
truth and origin-coded IDs and applies the requested common precision path.

## Backgrounds and precision

The new adapter reads the first M active rows directly from HDF5, without the
legacy endpoint's eta clipping or display rounding. Signed quantized mass codes
are matched to pi±, K±, p and pbar using nearest-code decoding. An aligned
`observed/p4` dataset is preferred after shape, finite-value, positive-energy,
mass-shell and packed-kinematic alignment checks. Original geometry and event
metadata are retained; source-only collision metadata is null.

Canonical files record thermal law/version, common input T, beta zero, acceptance,
species probabilities, root seed, RNG policy and emitted/accepted counts. These
metadata are included in the catalog, selected-event response and snapshot. New
files store aligned observed float32 p4 as well as the compatible packed point
cloud. Background/source proton sampling agrees exactly when masses, T and streams
are matched; normal production streams use separate namespaces.

Public file IDs hash the resolved path, so duplicate basenames remain distinct.
Pinned fingerprints use `stat-sha256-v1` over device, inode, size, mtime_ns and
ctime_ns, checked before and after reads and on cache hits/downloads. They detect
replacement and edits without hashing gigabytes on every drag. They are local
file-identity fingerprints, not portable content checksums: copying a file to a
different machine/path requires explicit catalog reselection. Fetching the catalog
again (or reloading the page) rescans allowed locations. Changed/missing files and
empty selections produce explicit errors. No alternate event is silently chosen.
Centrality is [min,max), except the final edge 1 is included. Original background
acceptance is reported as unknown when not specified; looser cuts cannot recover
tracks absent from a stored event.

**Native** uses fixed canonical component arrays, mixing stored background
precision with new float64 source p4. **Packed float16** composes each accepted
variant/composition separately, recomputes mean pT/eta, encodes all five relative
features as float16, stores event means as float32, then decodes. IDs and aligned
ordering survive this transform. Each composition has a separate observation
array and its diagnostics use that array. Existing background quantization cannot
be undone. Canonical components stay unchanged. Quantization is not a second cut.
Empty observations have empty arrays, zero counts/sums and null undefined means.

## Diagnostics, API and exports

Exact positive-proton diagnostics are deferred through a separate bounded pair
worker. Particle updates and static/catalog routes remain available. Pair work
uses 256×256 chunks without allocating the full O(n²) pair-vector array. Very
large proton samples can take time; the interface explicitly shows computation
in progress. No approximate histograms or invented detector scores are used.

`proton_clump_diagnostics_v1` defines uniform bins: pT 0–3 GeV / 60, y −2.5–2.5 /
50, wrapped Δφ −π–π / 72, |Δy| 0–5 / 50, q_inv 0–2 GeV / 80 and pair βT 0–1 /
50. All unordered observed proton pairs are counted once. Signed Δφ fills both
signs with half weight; raw counts, edges, weights, under/overflow and total
entries are returned. Density uses total entries including overflow. Green curves
show all observed positive protons (pT/y) or all observed proton pairs in the selected
composition; in combined mode this includes background–background, background–clump
and clump–clump pairs. Red curves show same-clump daughter pairs, explicitly
simulation truth. Bins are drawn at their actual edge positions as unfilled
step outlines: solid green and dashed red, so coincident histograms remain
distinguishable. The bin width is printed below each plot. Each histogram has its
own density normalization, and each plot has an explicit color legend. Fewer than two
protons gives “No proton pairs”. Diagnostic replies echo their complete view key
and cannot overwrite a newer scene or view.

| Route | Purpose |
| --- | --- |
| `GET /api/v1/capabilities` | Versions, supported features, service limits |
| `GET /api/v1/backgrounds` | Refreshed public catalog and scan warnings |
| `POST /api/v1/background/select` | Exact/random/next/previous selection and pinned reference |
| `POST /api/v1/scenario` | Validated deterministic bundle; echoes request_id |
| `POST /api/v1/diagnostics` | Exact deferred diagnostics for scenario/mode/variant/composition |
| `GET /api/v1/scenarios/{id}/recipe` | POSTable physical recipe with reproducibility versions |
| `GET /api/v1/scenarios/{id}/snapshot` | Full-resolution measured arrays, separate truth and provenance |

Recipe/snapshot downloads accept `observation_mode` and `variant` query parameters;
snapshots report `selected_variant`. All requested variants remain available in the snapshot.
The examples are `clump_explorer/examples/source_only.json` and
`clump_explorer/examples/variable_parameters.json`. Preview never writes HDF5 or
other persistent datasets. JSON request bodies are limited to 64 KiB and finite
numbers, unknown fields/modes are rejected, and requests must declare JSON.
Errors include field/code/message and request_id: malformed JSON 400, validation
422, unknown dataset/event 404, changed fingerprint 409, expired bundle 410,
bounded worker busy 503. A saved recipe regenerates an evicted scenario.

There are two generation workers and one pair worker. The LRU retains at most
32 bundles or 128 MiB of accounted Python containers including separately cached
observations/diagnostics. Snapshot and diagnostic views never replace canonical
physical arrays. Browser retries target only the latest draft/view.

## Verification

```bash
python -m unittest discover -s toys/event-display/clump_explorer/tests -v
node --test toys/event-display/clump_explorer/tests/test_browser_logic.mjs
python toys/event-display/clump_explorer/verify_preservation.py
python toys/event-display/serve_clump_events.py --benchmark 30
```

Python tests cover independent thermal radial integration, isotropy, mass shell,
zero/inverse boosts, q_inv invariance, truncated-Poisson conditioning, parameter
validation, stream coupling, control twins, multi-event β=0 isotropy, exact pair
chunks, edge cases, HDF5 precision/selection/fingerprints, 20,000 particles without
truncation, API errors/cache/exports, HTTP integration and batch/API parity.
Event-mixing tests additionally cover independent donor provenance, on-shell
four-vectors, ensemble one-body agreement, longitudinal pair-correlation removal,
control revisions, donor prefix preservation, packed ordering and recipe replay.
Node tests deliberately delay replies and cover latest-draft coalescing, reset,
auto-update off/Apply, failures/503 retry, fixed source coordinates and φ wrapping.
The SHA256 baseline captures all 21 pre-existing files, including initially
untracked work. It is a milestone audit, not a restriction on future intentional edits.

Measured on the development environment with NumPy 2.5.4, 30 uncached native
scenarios per case after two warmups, including generation, response/cache
preparation and JSON serialization; exact pair diagnostics are deferred:

| Case | Background tracks | p50 | p95 |
| --- | ---: | ---: | ---: |
| Source only | 0 | 0.92 ms | 1.25 ms |
| BES | 50 | 6.73 ms | 7.40 ms |
| Central LHC | 1,823 | 34.44 ms | 37.27 ms |

Live desktop/narrow-layout screenshots, browser keyboard checks, per-frame Canvas
timings and actual GPU allocation cleanup remain **unverified** in this environment:
the browser automation host is unavailable. The fallback and request-state code
is implemented, with unit checks for the race/coordinate contracts; these do not
replace a live browser inspection. Manual checks: drag controls rapidly, reset
during an update, toggle background/color/frame locally, compare fixed coordinates,
inspect both precision modes, resize to a narrow screen, tab through controls,
disable WebGPU/CDN, and repeat high-count updates while inspecting GPU resources.

This is a spinodal-inspired injection benchmark, not an exact parent decay or
globally conserving collision generator. It has no real flow/jets, CP probability,
ML detector, emission positions, radius or physical lifetime. Production HDF5
campaigns, exact recoil/conservation, detector response and scoring remain outside
this milestone.
