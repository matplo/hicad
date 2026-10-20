# Heavy-Ion Thermal Collision Point-Cloud Toy (`boltzmann`)

This package generates and validates synthetic heavy-ion collision datasets (`100,000` minimum-bias events per energy setting) in a standardized HDF5 point-cloud format designed for machine learning and statistical benchmarking across collision energies and Monte Carlo generators.

---

## 1. Repository Structure

| File | Description |
| :--- | :--- |
| `generate_thermal_h5.py` | Vectorized Monte Carlo generator for minimum-bias (`0–100%` centrality) thermal heavy-ion collision events with Glauber geometry ($b, N_{\mathrm{part}}, N_{\mathrm{coll}}$), Negative Binomial multiplicity fluctuations, radial-flow-boosted species mix ($\pi^\pm, K^\pm, p/\bar{p}$), and HDF5 `lzf`/`gzip` output. |
| `inspect_h5.py` | Fast CLI validator that checks HDF5 dataset shapes, exact dtypes (`float32` / `float16`), finite values, zero-padding integrity, and central vs. peripheral physics observables. |
| `verify_fits_and_plots.py` | Reconstructs single-particle kinematics ($p_T, \eta, \phi, Q\ln(1+m)$) from the `float16` point cloud, performs inclusive and species-resolved Boltzmann fits, saves multi-panel diagnostic figures to `plots/`, and generates the pre-executed Jupyter notebook `inspect_thermal_distributions.ipynb`. |
| `inspect_thermal_distributions.ipynb` | Pre-executed Jupyter notebook inspecting event-level Glauber distributions, reconstructed $p_T$, $\eta$, and $\phi$ spectra, and Boltzmann fit verification across all three collision energies. |
| `requirements.txt` | Pinned Python dependencies (`numpy`, `h5py`, `scipy`, `matplotlib`, `ipykernel`). |

---

## 2. HDF5 File Schema

Each generated `.h5` file contains three top-level datasets:

### 2.1 `mc_record` — `float32`, shape `(2,)`
Stores file-level Monte Carlo metadata:

| Index | Column | Type | Description |
| :---: | :--- | :---: | :--- |
| `0` | `Ecm` | `float32` | Center-of-mass energy per nucleon pair $\sqrt{s_{\mathrm{NN}}}$ [GeV] |
| `1` | `Label` | `float32` | Monte Carlo generator family & dataset code |

**MC Model Label Convention:**
- `0.0X` — Vanilla thermal events (Rita): `0.01` (Top LHC $5.02\text{ TeV}$), `0.02` (Top RHIC $200\text{ GeV}$), `0.03` (RHIC BES $9.2\text{ GeV}$)
- `0.1X` — UrQMD
- `0.2X` — Angantyr (PYTHIA 8 heavy-ion)
- `0.3X` — JEWEL
- `0.4X` — HIJING
- `0.5X` — AMPT

### 2.2 `event_record` — `float32`, shape `(N_events, 10)`
Stores event-by-event global geometry and kinematic summary observables:

| Index | Column | Type | Units / Range | Description |
| :---: | :--- | :---: | :---: | :--- |
| `0` | `centrality_class` | `float32` | `[0.0, 1.0]` | Centrality fraction (`0.0` = most central `0%`, `1.0` = `100%` peripheral) |
| `1` | `Multiplicity` | `float32` | integer $\ge 2$ | Charged particle multiplicity $M$ in $\|\eta\| < 1.0$ |
| `2` | `Impact_parameter` | `float32` | fm | Glauber impact parameter $b \propto \sqrt{c}\,b_{\max}$ |
| `3` | `N_part` | `float32` | $[2, 2A]$ | Number of participating nucleons $N_{\mathrm{part}}$ |
| `4` | `N_coll` | `float32` | $[1, N_{\mathrm{coll,max}}]$ | Number of binary nucleon-nucleon collisions $N_{\mathrm{coll}}$ |
| `5` | `sin(Psi2)` | `float32` | `[-1.0, 1.0]` | Event-plane sine $\sin(\Psi_2)$ |
| `6` | `cos(Psi2)` | `float32` | `[-1.0, 1.0]` | Event-plane cosine $\cos(\Psi_2)$ |
| `7` | `Mean_pT` | `float32` | GeV/$c$ | Event mean transverse momentum $\langle p_T \rangle_e = \frac{1}{M}\sum_{i=1}^M p_{T,i}$ |
| `8` | `Sum_pT` | `float32` | GeV/$c$ | Event total transverse momentum $\sum_{i=1}^M p_{T,i}$ |
| `9` | `Eta_mean` | `float32` | dimensionless | Event mean pseudorapidity $\langle \eta \rangle_e = \frac{1}{M}\sum_{i=1}^M \eta_i$ |

### 2.3 `point_cloud` — `float16`, shape `(N_events, max_M, 5)`
Zero-padded 3D array (`lzf` compressed by default). For an event $e$ with multiplicity $M = \text{int}(\text{event\_record}[e, 1])$, active particles occupy `point_cloud[e, :M, :]` and indices `M : max_M` are strictly `0.0`:

| Index | Column | Type | Reconstruction Formula | Description |
| :---: | :--- | :---: | :--- | :--- |
| `0` | `cos(phi)` | `float16` | $\phi_i = \operatorname{atan2}(\text{col}_1, \text{col}_0)$ | Cosine of particle azimuthal angle $\phi_i \in [-\pi, \pi)$ |
| `1` | `sin(phi)` | `float16` | $\phi_i = \operatorname{atan2}(\text{col}_1, \text{col}_0)$ | Sine of particle azimuthal angle $\phi_i \in [-\pi, \pi)$ |
| `2` | `log(pT/mean_pT)` | `float16` | $p_{T,i} = \langle p_T \rangle_e \cdot \exp(\text{col}_2)$ | Log-ratio of particle $p_{T,i}$ to event $\langle p_T \rangle_e$ |
| `3` | `Q * log(1 + mass)` | `float16` | $\|Q\ln(1+m)\| \in \{0.1307, 0.4012, 0.6618\}$ | Charge $Q_i \in \{\pm 1\}$ times $\ln(1 + m_i\,[\text{GeV}/c^2])$ for $\pi^\pm, K^\pm, p/\bar{p}$ |
| `4` | `eta_i - eta_mean` | `float16` | $\eta_i = \text{col}_4 + \langle \eta \rangle_e$ | Particle pseudorapidity $\eta_i \in (-1, 1)$ relative to event $\langle \eta \rangle_e$ |

---

## 3. Physics Settings & Generated Datasets (`100,000` Events / File)

Each particle's transverse momentum $p_T$ is sampled from a 2D thermal Boltzmann distribution ($\text{Gamma}(\text{shape}=2, \text{scale}=T_{\mathrm{eff},s})$):
$$\frac{dP}{dp_T} = A\,\frac{p_T}{T_{\mathrm{eff}}^2}\exp\!\left(-\frac{p_T}{T_{\mathrm{eff}}}\right), \qquad \ln\!\left(\frac{1}{p_T}\frac{dP}{dp_T}\right) = A_0 - \frac{p_T}{T_{\mathrm{eff}}}, \qquad \langle p_T \rangle = 2\,T_{\mathrm{eff}}$$
with radial-flow mass ordering $T_s \propto 1 + 0.65\,(m_s - m_\pi)$ across $\pi^\pm, K^\pm, p/\bar{p}$ calibrated so that the species-weighted central (`0–5%`) mean transverse momentum equals the target $\langle p_T \rangle_{0\text{–}5\%}$.

| Setting | System & $\sqrt{s_{\mathrm{NN}}}$ | MC Label | Target `0–5%` $\langle M \rangle$ ($\|\eta\|<1$) | Achieved `0–5%` $\langle M \rangle$ | Target `0–5%` $\langle p_T \rangle$ [GeV/$c$] | Achieved `0–5%` $\langle p_T \rangle$ [GeV/$c$] | Min-Bias $\langle M \rangle$ (`max_M`) | `point_cloud` Shape (`float16`) | File Size (`lzf`) |
| :--- | :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **1. Top LHC** | Pb–Pb $5020\text{ GeV}$ | `0.01` | `1940.0` | **1937.5** | `0.6800` | **0.6801** | `520.9` (`2486`) | `(100000, 2486, 5)` | `543.2 MB` |
| **2. Top RHIC** | Au–Au $200\text{ GeV}$ | `0.02` | `680.0` | **679.5** | `0.5600` | **0.5599** | `186.1` (`906`) | `(100000, 906, 5)` | `200.8 MB` |
| **3. RHIC BES** | Au–Au $9.2\text{ GeV}$ | `0.03` | `80.0` | **80.1** | `0.4200` | **0.4202** | `22.7` (`129`) | `(100000, 129, 5)` | `28.4 MB` |

---

## 4. Boltzmann Fit Verification Tables (`0–5%` Central Events)

### 4.1 Inclusive All-Charged $p_T$ Spectrum Fits
Because the inclusive charged hadron spectrum is a superposition of $\pi^\pm, K^\pm$, and $p/\bar{p}$ with $T_\pi < T_K < T_p$, a single-component PDF fit is weighted toward the low-$p_T$ pion peak while an invariant-yield logarithmic slope fit is weighted toward the heavier kaon/proton tail, bracketing the exact MLE effective temperature $\hat{T}_{\mathrm{MLE}} = \langle p_T \rangle / 2$:

| Dataset | Target $\langle p_T \rangle$ [GeV/$c$] | Reconstructed Point-Cloud $\langle p_T \rangle$ [GeV/$c$] | Target $T_{\mathrm{eff}}$ [GeV] | MLE $\hat{T}_{\mathrm{eff}} = \langle p_T \rangle / 2$ [GeV] | Single-Boltzmann PDF Fit $T_{\mathrm{eff}}$ ($2T_{\mathrm{eff}}$) | Invariant Log-Slope Fit $T_{\mathrm{eff}}$ ($2T_{\mathrm{eff}}$) |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| **Top LHC ($5.02\text{ TeV}$, `0.01`)** | `0.6800` | **0.6802** | `0.3400` | **0.3401** | `0.3362 GeV` (`0.6724 GeV/c`) | `0.3430 GeV` (`0.6860 GeV/c`) |
| **Top RHIC ($200\text{ GeV}$, `0.02`)** | `0.5600` | **0.5600** | `0.2800` | **0.2800** | `0.2764 GeV` (`0.5529 GeV/c`) | `0.2873 GeV` (`0.5746 GeV/c`) |
| **RHIC BES ($9.2\text{ GeV}$, `0.03`)** | `0.4200` | **0.4205** | `0.2100` | **0.2103** | `0.2058 GeV` (`0.4116 GeV/c`) | `0.2290 GeV` (`0.4581 GeV/c`) |

### 4.2 Identified Hadron Species ($\pi^\pm, K^\pm, p/\bar{p}$) Boltzmann Fits
When particles are separated by $|Q\ln(1+m)|$ (`point_cloud[:, :, 3]`) and fitted individually with `boltzmann_pdf`, each hadron species follows an **exact** Boltzmann distribution and the species-weighted sum $\sum_s f_s (2 T_s^{\mathrm{fit}})$ reproduces the exact target $\langle p_T \rangle$:

| Dataset | Species | Fraction $f_s$ | Target $T_s$ [GeV] | Fitted $T_s$ [GeV] | Target $\langle p_T \rangle_s$ [GeV/$c$] | Fitted $2\,T_s$ [GeV/$c$] |
| :--- | :--- | :---: | :---: | :---: | :---: | :---: |
| **Top LHC ($5.02\text{ TeV}$)** | Pions ($\pi^\pm$) | `0.830` | `0.3232` | **0.3238** | `0.6463` | **0.6475** |
| | Kaons ($K^\pm$) | `0.124` | `0.3975` | **0.3973** | `0.7951` | **0.7947** |
| | Protons ($p,\bar{p}$) | `0.046` | `0.4909` | **0.4904** | `0.9818` | **0.9808** |
| | **Species-Weighted Total** | `1.000` | `0.3400` | **0.3405** | `0.6800` | **0.6811** |
| **Top RHIC ($200\text{ GeV}$)** | Pions ($\pi^\pm$) | `0.810` | `0.2639` | **0.2646** | `0.5278` | **0.5292** |
| | Kaons ($K^\pm$) | `0.130` | `0.3246` | **0.3250** | `0.6492` | **0.6501** |
| | Protons ($p,\bar{p}$) | `0.060` | `0.4009` | **0.4025** | `0.8018` | **0.8051** |
| | **Species-Weighted Total** | `1.000` | `0.2800` | **0.2807** | `0.5600` | **0.5614** |
| **RHIC BES ($9.2\text{ GeV}$)** | Pions ($\pi^\pm$) | `0.740` | `0.1893` | **0.1921** | `0.3786` | **0.3843** |
| | Kaons ($K^\pm$) | `0.120` | `0.2329` | **0.2352** | `0.4658` | **0.4705** |
| | Protons ($p,\bar{p}$) | `0.139` | `0.2876` | **0.2910** | `0.5752` | **0.5820** |
| | **Species-Weighted Total** | `1.000` | `0.2100` | **0.2111** | `0.4200` | **0.4222** |

---

## 5. Quick Start

### Using `henv -x`
```bash
# 1. Generate all three 100,000-event HDF5 files into output/
henv -x python generate_thermal_h5.py --setting all --n-events 100000 --output-dir output

# 2. Validate HDF5 schema, dtypes, zero-padding, and central/peripheral averages
henv -x python inspect_h5.py output

# 3. Run Boltzmann fits, save PNG plots to plots/, and build inspect_thermal_distributions.ipynb
henv -x python verify_fits_and_plots.py
```

### Using a local `.venv`
```bash
python3 -m venv .venv
.venv/bin/pip install -r requirements.txt

.venv/bin/python generate_thermal_h5.py --setting all --n-events 100000 --output-dir output
.venv/bin/python inspect_h5.py output
.venv/bin/python verify_fits_and_plots.py
```
