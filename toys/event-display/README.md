# 3D HDF5 Point-Cloud Event Display (`event-display`)

A lightweight Python HTTP streaming server ([`serve_h5_events.py`](serve_h5_events.py)) paired with a Three.js WebGPU + TSL 3D event display ([`h5_event_display.html`](h5_event_display.html)) styled after `visual-eds`.

Designed purely for **3D momentum-space / point-cloud visualization** of heavy-ion HDF5 datasets:
- **No detector geometry** (no barrels, TPC cages, silicon staves, or endcaps)
- **No magnetic field** ($B = 0$ — straight-line momentum rays and 3D point-cloud markers)
- **No fireball / QGP volume** and **no Glauber spectator/participant nucleons**

---

## Files

| File | Description |
| :--- | :--- |
| [`serve_h5_events.py`](serve_h5_events.py) | Multi-threaded Python HTTP server (`h5py` + `numpy`) that indexes `event_record` in memory for instant centrality filtering and lazily slices single events from `point_cloud` on demand |
| [`h5_event_display.html`](h5_event_display.html) | Standalone Three.js WebGPU (`three@0.171.0`, `three/tsl`, `OrbitControls`, `BloomNode`) 3D interactive viewer with `Barlow Condensed` + `IBM Plex Mono` HUD, centrality filter, and live TSL shader animation |

---

## Quick Start

Run the streaming server from the `event-display` directory using `henv -x`:

```bash
henv -x python serve_h5_events.py --open-browser
```

By default, the server binds to `http://127.0.0.1:8765/` and automatically searches for `.h5` files in:
1. `--data-dir` (if provided)
2. `./output/`
3. `../boltzmann/output/` and `../boltzmann/`

To point to a specific directory or `.h5` file on a custom port:

```bash
henv -x python serve_h5_events.py --data-dir ../boltzmann/output --port 8765
```

---

## HTTP Streaming API

| Endpoint | Query Parameters | Description |
| :--- | :--- | :--- |
| `GET /` | — | Serves [`h5_event_display.html`](h5_event_display.html) |
| `GET /api/files` | — | Lists discovered `.h5` files with `mc_record` metadata ($E_{\text{cm}}$, MC model label, event count, $M_{\max}$, $\langle M \rangle$, $\langle p_T \rangle$) |
| `GET /api/event` | `file`, `idx`, `nav` (`exact` \| `random` \| `next` \| `prev`), `cur_idx`, `c_min`, `c_max` | Lazily reads `point_cloud[idx, :multiplicity, :]` from disk, unpacks the 5 `float16` features using the event's `event_record`, and returns JSON |

---

## 3D Reconstruction from HDF5 Point-Cloud Features

For each active particle $i \in [0, M-1]$ in an event, the server unpacks the 5 `float16` columns of `point_cloud` together with the event's `event_record`:

| `point_cloud` Column | Stored Feature (`float16`) | Unpacked Physical Quantity |
| :---: | :--- | :--- |
| `0`, `1` | $\cos\phi_i,\; \sin\phi_i$ | Azimuthal direction $\hat{n}_{\perp,i} = (\cos\phi_i, \sin\phi_i)$ (re-normalized on unit circle) |
| `2` | $\ln(p_{T,i} / \langle p_T \rangle_{\text{ev}})$ | Transverse momentum $p_{T,i} = \langle p_T \rangle_{\text{ev}} \exp(\text{col}_2)$ $[\text{GeV}/c]$ |
| `3` | $Q_i \ln(1 + m_i)$ | Charge $Q_i = \text{sign}(\text{col}_3) \in \{-1, +1\}$ and species $\pi^\pm, K^\pm, p/\bar{p}$ via nearest $\ln(1+m)$ lookup |
| `4` | $\eta_i - \langle \eta \rangle_{\text{ev}}$ | Pseudorapidity $\eta_i = \text{col}_4 + \langle \eta \rangle_{\text{ev}}$ |

In 3D scene coordinates (where the $z$-axis is the beam axis and $(x, y)$ is the transverse plane):
- **3D unit direction**:
  $$\hat{u}_i = \left(\frac{\cos\phi_i}{\cosh\eta_i},\; \frac{\sin\phi_i}{\cosh\eta_i},\; \tanh\eta_i\right)$$
- **Geometry Modes**:
  1. **Rays + Cloud ($p_T$ length)**: Straight momentum rays + 3D octahedron point-cloud markers at radius $R_i \propto \ln(1 + p_{T,i} / 0.35\text{ GeV})$.
  2. **Point Cloud Only (3D $\vec{p}$)**: Pure 3D point-cloud markers without ray segments.
  3. **Unit Sphere ($\beta = p/E$ flight)**: Particles propagate outward from the origin at their relativistic velocity $\beta_i = p_i / \sqrt{p_i^2 + m_i^2}$, naturally separating $\pi^\pm$, $K^\pm$, and $p/\bar{p}$ time-of-flight shells during the expansion animation.
- **Colour Modes**:
  - `Transverse Momentum pT` (cyan $\to$ blue $\to$ magenta $\to$ amber-white)
  - `H5 Feature: log(pT / <pT>)` (diverging blue $\to$ teal $\to$ gold-red around $\langle p_T \rangle_{\text{ev}}$)
  - `Particle Species (π / K / p)` (decoded from $|Q\ln(1+m)|$)
  - `Electric Charge (Q+ / Q−)` (sign of $Q\ln(1+m)$)
  - `Pseudorapidity η`
