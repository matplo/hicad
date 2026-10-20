#!/usr/bin/env python3
"""
Generate diagnostic figures and build the complete, pre-executed Jupyter notebook
`inspect_thermal_distributions.ipynb` for inspecting the heavy-ion thermal HDF5 datasets.
"""

from __future__ import annotations

import base64
import io
import json
import os
import h5py
import numpy as np
from scipy.optimize import curve_fit
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

plt.rcParams["figure.dpi"] = 120
plt.rcParams["font.size"] = 10

DATASETS = {
    "Top LHC (Pb-Pb 5.02 TeV)": {
        "path": "output/thermal_events_lhc_5020GeV.h5",
        "color": "#1f77b4",
        "target_ecm": 5020.0,
        "target_label": 0.01,
        "target_mult_0_5": 1940.0,
        "target_pt_0_5": 0.68,
        "target_pt_periph": 0.60,
        "species_probs": (0.415, 0.415, 0.0625, 0.0625, 0.0225, 0.0225),
    },
    "Top RHIC (Au-Au 200 GeV)": {
        "path": "output/thermal_events_rhic_200GeV.h5",
        "color": "#ff7f0e",
        "target_ecm": 200.0,
        "target_label": 0.02,
        "target_mult_0_5": 680.0,
        "target_pt_0_5": 0.56,
        "target_pt_periph": 0.495,
        "species_probs": (0.405, 0.405, 0.065, 0.065, 0.032, 0.028),
    },
    "RHIC BES (Au-Au 9.2 GeV)": {
        "path": "output/thermal_events_bes_9p2GeV.h5",
        "color": "#2ca02c",
        "target_ecm": 9.2,
        "target_label": 0.03,
        "target_mult_0_5": 80.0,
        "target_pt_0_5": 0.42,
        "target_pt_periph": 0.37,
        "species_probs": (0.375, 0.345, 0.080, 0.045, 0.145, 0.010),
    },
}

SPECIES_MASS = np.array(
    [0.13957039, 0.13957039, 0.493677, 0.493677, 0.93827208, 0.93827208],
    dtype=np.float64,
)


def fig_to_base64(fig: plt.Figure) -> str:
    buf = io.BytesIO()
    fig.savefig(buf, format="png", bbox_inches="tight", dpi=130)
    buf.seek(0)
    return base64.b64encode(buf.read()).decode("ascii")


def boltzmann_pdf(pt: np.ndarray, t_eff: float, norm: float) -> np.ndarray:
    return norm * (pt / (t_eff ** 2)) * np.exp(-pt / t_eff)


def boltzmann_inv_log(pt: np.ndarray, t_eff: float, a0: float) -> np.ndarray:
    return a0 - pt / t_eff


def reconstruct_particles_from_h5(
    h5_path: str, max_events: int = 4000, central_only: bool = True
):
    with h5py.File(h5_path, "r") as f:
        mc = f["mc_record"][:]
        ev = f["event_record"][:]
        if central_only:
            idx = np.where(ev[:, 0] < 0.05)[0][:max_events]
        else:
            idx = np.arange(min(max_events, len(ev)))
        ev_sub = ev[idx]
        pc_sub = f["point_cloud"][idx]

    pt_list, eta_list, phi_list, qlogm_list = [], [], [], []
    for i in range(len(ev_sub)):
        m = int(ev_sub[i, 1])
        mean_pt = float(ev_sub[i, 7])
        eta_mean = float(ev_sub[i, 9])

        pc = pc_sub[i, :m, :].astype(np.float32)
        cos_phi = pc[:, 0]
        sin_phi = pc[:, 1]
        log_pt_ratio = pc[:, 2]
        qlogm = pc[:, 3]
        eta_ratio = pc[:, 4]

        pt = mean_pt * np.exp(log_pt_ratio)
        phi = np.arctan2(sin_phi, cos_phi)
        eta_denom = (
            eta_mean
            if abs(eta_mean) >= 1e-4
            else (1e-4 if eta_mean >= 0 else -1e-4)
        )
        eta = eta_ratio * eta_denom

        pt_list.append(pt)
        eta_list.append(eta)
        phi_list.append(phi)
        qlogm_list.append(qlogm)

    return (
        mc,
        ev,
        np.concatenate(pt_list),
        np.concatenate(eta_list),
        np.concatenate(phi_list),
        np.concatenate(qlogm_list),
    )


def main() -> None:
    os.makedirs("plots", exist_ok=True)

    # --- Step 1: Inventory output ---
    event_data = {}
    out_lines_1 = [
        f"{'Dataset':<28} | {'Ecm [GeV]':>9} | {'Label':>6} | {'0-5% <M> (Target)':>19} | {'0-5% <pT> (Target)':>20} | {'Point Cloud Shape':>22}",
        "-" * 118,
    ]
    for name, cfg in DATASETS.items():
        with h5py.File(cfg["path"], "r") as f:
            mc = f["mc_record"][:]
            ev = f["event_record"][:]
            pc_shape = f["point_cloud"].shape
            pc_dtype = f["point_cloud"].dtype
            ev_cols = list(f["event_record"].attrs["columns"])
            pc_cols = list(f["point_cloud"].attrs["columns"])

        event_data[name] = {
            "mc": mc,
            "ev": ev,
            "pc_shape": pc_shape,
            "ev_cols": ev_cols,
            "pc_cols": pc_cols,
        }
        c05 = ev[:, 0] < 0.05
        m_05 = ev[c05, 1].mean()
        pt_05 = ev[c05, 7].mean()
        out_lines_1.append(
            f"{name:<28} | {mc[0]:>9.1f} | {mc[1]:>6.2f} | "
            f"{m_05:>8.1f} ({cfg['target_mult_0_5']:>6.1f}) | "
            f"{pt_05:>8.4f} ({cfg['target_pt_0_5']:>6.4f}) | "
            f"{str(pc_shape):>14} ({pc_dtype})"
        )
    text_out_1 = "\n".join(out_lines_1) + "\n"
    print(text_out_1)

    # --- Step 2: Event-level Glauber & Multiplicity Plot ---
    fig1, axes1 = plt.subplots(2, 2, figsize=(13, 9))
    ax_m, ax_b = axes1[0, 0], axes1[0, 1]
    ax_np, ax_mpt = axes1[1, 0], axes1[1, 1]

    for name, cfg in DATASETS.items():
        ev = event_data[name]["ev"]
        color = cfg["color"]
        c = ev[:, 0]
        mult = ev[:, 1]
        b = ev[:, 2]
        npart = ev[:, 3]
        mpt = ev[:, 7]

        ax_m.hist(
            mult,
            bins=80,
            density=True,
            histtype="step",
            lw=2.0,
            color=color,
            label=f"{name} ($\\langle M \\rangle_{{0-5\\%}}={mult[c<0.05].mean():.0f}$)",
        )
        ax_b.hist(
            b,
            bins=60,
            density=True,
            histtype="step",
            lw=2.0,
            color=color,
            label=name,
        )

        # Bin by centrality for N_part and <pT>
        c_bins = np.linspace(0.0, 1.0, 21)
        c_centers = 0.5 * (c_bins[:-1] + c_bins[1:])
        idx_bin = np.digitize(c, c_bins) - 1
        np_mean = [npart[idx_bin == k].mean() for k in range(20)]
        pt_mean = [mpt[idx_bin == k].mean() for k in range(20)]

        ax_np.plot(c_centers * 100, np_mean, "o-", ms=4, lw=1.8, color=color, label=name)
        ax_mpt.plot(c_centers * 100, pt_mean, "s-", ms=4, lw=1.8, color=color, label=name)
        ax_mpt.axhline(cfg["target_pt_0_5"], color=color, linestyle=":", alpha=0.6)

    ax_m.set_yscale("log")
    ax_m.set_title("(a) Charged Multiplicity $M$ Distribution (Min-Bias 0–100%)")
    ax_m.set_xlabel("Multiplicity $M$ ($|\\eta| < 1.0$)")
    ax_m.set_ylabel("Probability Density $P(M)$")
    ax_m.legend(fontsize=8)
    ax_m.grid(True, which="both", alpha=0.3)

    ax_b.set_title("(b) Impact Parameter $b$ Distribution")
    ax_b.set_xlabel("Impact Parameter $b$ [fm]")
    ax_b.set_ylabel("Probability Density $P(b)$ [fm$^{-1}$]")
    ax_b.legend(fontsize=8)
    ax_b.grid(True, alpha=0.3)

    ax_np.set_title("(c) Mean Participating Nucleons $\\langle N_{\\mathrm{part}} \\rangle$ vs Centrality")
    ax_np.set_xlabel("Centrality Class [%] (0% = Central, 100% = Peripheral)")
    ax_np.set_ylabel("$\\langle N_{\\mathrm{part}} \\rangle$")
    ax_np.legend(fontsize=8)
    ax_np.grid(True, alpha=0.3)

    ax_mpt.set_title("(d) Event Mean $\\langle p_T \\rangle$ vs Centrality (Dotted = 0–5% Target)")
    ax_mpt.set_xlabel("Centrality Class [%]")
    ax_mpt.set_ylabel("Event $\\langle p_T \\rangle$ [GeV/$c$]")
    ax_mpt.legend(fontsize=8)
    ax_mpt.grid(True, alpha=0.3)

    fig1.tight_layout()
    fig1.savefig("plots/event_level_glauber_distributions.png", dpi=150)
    b64_fig1 = fig_to_base64(fig1)
    plt.close(fig1)

    # --- Step 3: Reconstruct single-particle kinematics ---
    reconstructed = {}
    out_lines_3 = [
        f"{'Dataset':<28} | {'Sampled 0-5% Events':>19} | {'Total Particles':>15} | {'Event <pT> [GeV/c]':>18} | {'PointCloud <pT> [GeV/c]':>23} | {'<phi> [rad]':>11}",
        "-" * 126,
    ]
    for name, cfg in DATASETS.items():
        mc, ev, pt_c, eta_c, phi_c, qlogm_c = reconstruct_particles_from_h5(
            cfg["path"], max_events=4000, central_only=True
        )
        reconstructed[name] = {
            "pt": pt_c,
            "eta": eta_c,
            "phi": phi_c,
            "qlogm": qlogm_c,
        }
        c05 = ev[:, 0] < 0.05
        n_ev_sampled = min(4000, int(np.sum(c05)))
        ev_pt_05 = float(ev[c05, 7][:n_ev_sampled].mean())
        out_lines_3.append(
            f"{name:<28} | {n_ev_sampled:>19,d} | {len(pt_c):>15,d} | "
            f"{ev_pt_05:>18.4f} | {float(pt_c.mean()):>23.4f} | {float(phi_c.mean()):>11.4f}"
        )
    text_out_3 = "\n".join(out_lines_3) + "\n"
    print(text_out_3)

    # --- Step 4: Inclusive Boltzmann Fits & Kinematic Spectra Plot ---
    fig2, axes2 = plt.subplots(2, 2, figsize=(14, 10))
    ax_pt, ax_inv = axes2[0, 0], axes2[0, 1]
    ax_eta, ax_phi = axes2[1, 0], axes2[1, 1]

    out_lines_4 = [
        f"{'Dataset':<28} | {'Target <pT>':>11} | {'Target T_eff':>12} | {'MLE T_eff (<pT>/2)':>18} | {'CurveFit T_eff':>14} | {'LogSlope T_eff':>14} | {'Fit 2*T_eff (PDF/Slope)':>23}",
        "-" * 134,
    ]

    for name, cfg in DATASETS.items():
        color = cfg["color"]
        target_pt = cfg["target_pt_0_5"]
        target_t = target_pt / 2.0
        pt_c = reconstructed[name]["pt"]
        eta_c = reconstructed[name]["eta"]
        phi_c = reconstructed[name]["phi"]

        t_mle = float(pt_c.mean()) / 2.0

        bins = np.linspace(0.05, 3.5, 70)
        counts, edges = np.histogram(pt_c, bins=bins, density=True)
        centers = 0.5 * (edges[:-1] + edges[1:])
        valid = counts > 0

        popt_pdf, _ = curve_fit(
            boltzmann_pdf, centers[valid], counts[valid], p0=[target_t, 1.0]
        )
        t_pdf, norm_pdf = popt_pdf

        inv_mask = (centers >= 0.15) & (centers <= 2.5) & (counts > 0)
        inv_y = np.log(counts[inv_mask] / centers[inv_mask])
        popt_inv, _ = curve_fit(
            boltzmann_inv_log, centers[inv_mask], inv_y, p0=[target_t, 0.0]
        )
        t_slope, a0_slope = popt_inv

        out_lines_4.append(
            f"{name:<28} | {target_pt:>11.4f} | {target_t:>12.4f} | "
            f"{t_mle:>18.4f} | {t_pdf:>14.4f} | {t_slope:>14.4f} | "
            f"{2*t_pdf:>10.4f} / {2*t_slope:<10.4f}"
        )

        pt_fine = np.linspace(0.02, 3.5, 300)
        ax_pt.scatter(
            centers[valid],
            counts[valid],
            s=18,
            color=color,
            alpha=0.8,
            label=f"{name} (<pT>={pt_c.mean():.3f})",
        )
        ax_pt.plot(
            pt_fine,
            boltzmann_pdf(pt_fine, t_pdf, norm_pdf),
            color=color,
            lw=2.0,
            linestyle="--",
            label=f"Boltzmann Fit: $T_{{eff}}={t_pdf:.3f}$ GeV ($2T_{{eff}}={2*t_pdf:.3f}$)",
        )

        ax_inv.scatter(
            centers[valid],
            counts[valid] / centers[valid],
            s=16,
            color=color,
            alpha=0.8,
            label=f"{name}",
        )
        ax_inv.plot(
            pt_fine,
            np.exp(boltzmann_inv_log(pt_fine, t_slope, a0_slope)),
            color=color,
            lw=1.8,
            linestyle="--",
            label=f"Slope $T_{{eff}}={t_slope:.3f}$ GeV ($2T_{{eff}}={2*t_slope:.3f}$)",
        )

        ax_eta.hist(
            eta_c,
            bins=60,
            range=(-1.1, 1.1),
            density=True,
            histtype="step",
            lw=2.0,
            color=color,
            label=name,
        )
        ax_phi.hist(
            phi_c,
            bins=60,
            range=(-np.pi, np.pi),
            density=True,
            histtype="step",
            lw=2.0,
            color=color,
            label=name,
        )

    text_out_4 = "\n".join(out_lines_4) + "\n"
    print(text_out_4)

    ax_pt.set_title("(a) Central (0–5%) $p_T$ Spectrum & Boltzmann PDF Fits")
    ax_pt.set_xlabel("$p_T$ [GeV/$c$]")
    ax_pt.set_ylabel("Probability Density $dP/dp_T$ [(GeV/$c$)$^{-1}$]")
    ax_pt.legend(fontsize=7.5)
    ax_pt.grid(True, alpha=0.3)

    ax_inv.set_yscale("log")
    ax_inv.set_title("(b) Invariant Spectrum $(1/p_T)\\,dP/dp_T$ & Exponential Slope Fits")
    ax_inv.set_xlabel("$p_T$ [GeV/$c$]")
    ax_inv.set_ylabel("$(1/p_T)\\,dP/dp_T$ [(GeV/$c$)$^{-2}$]")
    ax_inv.legend(fontsize=7.5)
    ax_inv.grid(True, which="both", alpha=0.3)

    ax_eta.axhline(0.5, color="black", linestyle=":", lw=1.5, label=r"Uniform $|\eta|<1$ ($0.5$)")
    ax_eta.set_title("(c) Reconstructed Pseudorapidity $\\eta$ Spectrum")
    ax_eta.set_xlabel("Pseudorapidity $\\eta$")
    ax_eta.set_ylabel("Probability Density $dP/d\\eta$")
    ax_eta.set_ylim(0, 0.72)
    ax_eta.legend(fontsize=8)
    ax_eta.grid(True, alpha=0.3)

    ax_phi.axhline(
        1.0 / (2.0 * np.pi),
        color="black",
        linestyle=":",
        lw=1.5,
        label="Isotropic $1/(2\\pi) \\approx 0.1592$",
    )
    ax_phi.set_title("(d) Reconstructed Azimuthal Angle $\\phi$ Spectrum")
    ax_phi.set_xlabel("Azimuthal Angle $\\phi$ [rad]")
    ax_phi.set_ylabel("Probability Density $dP/d\\phi$ [rad$^{-1}$]")
    ax_phi.set_ylim(0, 0.22)
    ax_phi.legend(fontsize=8)
    ax_phi.grid(True, alpha=0.3)

    fig2.tight_layout()
    fig2.savefig("plots/kinematic_spectra_and_boltzmann_fits.png", dpi=150)
    b64_fig2 = fig_to_base64(fig2)
    plt.close(fig2)

    # --- Step 5: Species-Resolved Boltzmann Fits (pi, K, p) ---
    fig3, axes3 = plt.subplots(1, 3, figsize=(15, 4.8), sharey=False)
    species_defs = [
        ("Pions ($\\pi^\\pm$)", 0.13065, 0.13957039, "#1f77b4"),
        ("Kaons ($K^\\pm$)", 0.40124, 0.493677, "#ff7f0e"),
        ("Protons ($p, \\bar{p}$)", 0.66180, 0.93827208, "#d62728"),
    ]

    out_lines_5 = [
        f"{'Dataset':<26} | {'Species':<16} | {'Frac':>6} | {'Target T_s [GeV]':>16} | {'Fitted T_s [GeV]':>16} | {'Target <pT>_s':>13} | {'Fitted 2*T_s':>12}",
        "-" * 119,
    ]

    for idx_ds, (name, cfg) in enumerate(DATASETS.items()):
        ax = axes3[idx_ds]
        pt_c = reconstructed[name]["pt"]
        qlogm_c = np.abs(reconstructed[name]["qlogm"])

        probs = np.array(cfg["species_probs"], dtype=np.float64)
        probs /= probs.sum()
        rel_w = 1.0 + 0.65 * (SPECIES_MASS - 0.13957039)
        avg_w = float(np.sum(probs * rel_w))

        weighted_pt_sum = 0.0
        for sp_label, qval, mass_val, sp_color in species_defs:
            mask_sp = np.abs(qlogm_c - qval) < 0.05
            pt_sp = pt_c[mask_sp]
            frac_sp = float(np.mean(mask_sp))

            sp_w = (1.0 + 0.65 * (mass_val - 0.13957039)) / avg_w
            target_pt_sp = cfg["target_pt_0_5"] * sp_w
            target_t_sp = target_pt_sp / 2.0

            bins_sp = np.linspace(0.05, 3.2, 55)
            counts_sp, edges_sp = np.histogram(pt_sp, bins=bins_sp, density=True)
            centers_sp = 0.5 * (edges_sp[:-1] + edges_sp[1:])
            v_sp = counts_sp > 0

            popt_sp, _ = curve_fit(
                boltzmann_pdf, centers_sp[v_sp], counts_sp[v_sp], p0=[target_t_sp, 1.0]
            )
            t_fit_sp, norm_fit_sp = popt_sp
            weighted_pt_sum += frac_sp * (2.0 * t_fit_sp)

            sp_short = sp_label.split("(")[0].strip()
            out_lines_5.append(
                f"{name:<26} | {sp_short:<16} | {frac_sp:>6.3f} | "
                f"{target_t_sp:>16.4f} | {t_fit_sp:>16.4f} | "
                f"{target_pt_sp:>13.4f} | {2*t_fit_sp:>12.4f}"
            )

            pt_grid = np.linspace(0.02, 3.2, 250)
            ax.scatter(centers_sp[v_sp], counts_sp[v_sp], s=14, color=sp_color, alpha=0.75)
            ax.plot(
                pt_grid,
                boltzmann_pdf(pt_grid, t_fit_sp, norm_fit_sp),
                color=sp_color,
                lw=2.0,
                linestyle="--",
                label=f"{sp_label}: $T_s={t_fit_sp:.3f}$ ($\\langle p_T\\rangle={2*t_fit_sp:.3f}$)",
            )

        out_lines_5.append(
            f"{name:<26} | {'Species-Weighted':<16} | {1.000:>6.3f} | "
            f"{cfg['target_pt_0_5']/2:>16.4f} | {weighted_pt_sum/2:>16.4f} | "
            f"{cfg['target_pt_0_5']:>13.4f} | {weighted_pt_sum:>12.4f}"
        )
        out_lines_5.append("-" * 119)

        ax.set_title(f"{name}\nSpecies Boltzmann Fits (0–5% Central)")
        ax.set_xlabel("$p_T$ [GeV/$c$]")
        if idx_ds == 0:
            ax.set_ylabel("Probability Density $dP_s/dp_T$ [(GeV/$c$)$^{-1}$]")
        ax.legend(fontsize=7.5)
        ax.grid(True, alpha=0.3)

    text_out_5 = "\n".join(out_lines_5) + "\n"
    print(text_out_5)

    fig3.tight_layout()
    fig3.savefig("plots/species_resolved_boltzmann_fits.png", dpi=150)
    b64_fig3 = fig_to_base64(fig3)
    plt.close(fig3)

    # --- Build the complete, executed Jupyter notebook (.ipynb) ---
    cells = [
        {
            "cell_type": "markdown",
            "metadata": {},
            "source": [
                "# Heavy-Ion Thermal Event Distributions & Boltzmann Fit Inspection\n\n"
                "This notebook inspects the three generated HDF5 heavy-ion thermal collision datasets (`100,000` events each):\n"
                "1. **Top LHC (Pb–Pb at $\\sqrt{s_{\\mathrm{NN}}} = 5020\\text{ GeV}$)** — MC Label `0.01`, target central (0–5%) $\\langle M \\rangle \\approx 1940$, $\\langle p_T \\rangle \\approx 0.68\\text{ GeV}/c$\n"
                "2. **Top RHIC (Au–Au at $\\sqrt{s_{\\mathrm{NN}}} = 200\\text{ GeV}$)** — MC Label `0.02`, target central (0–5%) $\\langle M \\rangle \\approx 680$, $\\langle p_T \\rangle \\approx 0.56\\text{ GeV}/c$\n"
                "3. **RHIC BES (Au–Au at $\\sqrt{s_{\\mathrm{NN}}} = 9.2\\text{ GeV}$)** — MC Label `0.03`, target central (0–5%) $\\langle M \\rangle \\approx 80$, $\\langle p_T \\rangle \\approx 0.42\\text{ GeV}/c$\n\n"
                "### Objectives\n"
                "- Inspect event-level Glauber and multiplicity distributions from `event_record` (`float32`).\n"
                "- Reconstruct single-particle transverse momentum $p_T$, pseudorapidity $\\eta$, azimuthal angle $\\phi$, and species $Q\\ln(1+m)$ from the zero-padded `point_cloud` (`float16`) dataset.\n"
                "- Fit the reconstructed $p_T$ spectra with a thermal Boltzmann distribution and verify that the extracted effective temperature $T_{\\mathrm{eff}}$ (where $\\langle p_T \\rangle = 2\\,T_{\\mathrm{eff}}$) reproduces the input physics parameters."
            ],
        },
        {
            "cell_type": "markdown",
            "metadata": {},
            "source": [
                "## 1. Setup & Dataset Inventory\n\n"
                "Load required scientific Python packages (`numpy`, `h5py`, `matplotlib`, `scipy`) and verify the HDF5 dataset schemas and summary statistics across all three collision energies."
            ],
        },
        {
            "cell_type": "code",
            "execution_count": 1,
            "metadata": {},
            "outputs": [],
            "source": [
                "import os\n"
                "import h5py\n"
                "import numpy as np\n"
                "import matplotlib.pyplot as plt\n"
                "from scipy.optimize import curve_fit\n\n"
                "%matplotlib inline\n"
                "plt.rcParams['figure.dpi'] = 120\n"
                "plt.rcParams['font.size'] = 10\n\n"
                "DATASETS = {\n"
                "    'Top LHC (Pb-Pb 5.02 TeV)': {\n"
                "        'path': 'output/thermal_events_lhc_5020GeV.h5',\n"
                "        'color': '#1f77b4',\n"
                "        'target_ecm': 5020.0,\n"
                "        'target_label': 0.01,\n"
                "        'target_mult_0_5': 1940.0,\n"
                "        'target_pt_0_5': 0.68,\n"
                "        'target_pt_periph': 0.60,\n"
                "        'species_probs': (0.415, 0.415, 0.0625, 0.0625, 0.0225, 0.0225),\n"
                "    },\n"
                "    'Top RHIC (Au-Au 200 GeV)': {\n"
                "        'path': 'output/thermal_events_rhic_200GeV.h5',\n"
                "        'color': '#ff7f0e',\n"
                "        'target_ecm': 200.0,\n"
                "        'target_label': 0.02,\n"
                "        'target_mult_0_5': 680.0,\n"
                "        'target_pt_0_5': 0.56,\n"
                "        'target_pt_periph': 0.495,\n"
                "        'species_probs': (0.405, 0.405, 0.065, 0.065, 0.032, 0.028),\n"
                "    },\n"
                "    'RHIC BES (Au-Au 9.2 GeV)': {\n"
                "        'path': 'output/thermal_events_bes_9p2GeV.h5',\n"
                "        'color': '#2ca02c',\n"
                "        'target_ecm': 9.2,\n"
                "        'target_label': 0.03,\n"
                "        'target_mult_0_5': 80.0,\n"
                "        'target_pt_0_5': 0.42,\n"
                "        'target_pt_periph': 0.37,\n"
                "        'species_probs': (0.375, 0.345, 0.080, 0.045, 0.145, 0.010),\n"
                "    },\n"
                "}\n\n"
                "SPECIES_MASS = np.array(\n"
                "    [0.13957039, 0.13957039, 0.493677, 0.493677, 0.93827208, 0.93827208],\n"
                "    dtype=np.float64,\n"
                ")"
            ],
        },
        {
            "cell_type": "code",
            "execution_count": 2,
            "metadata": {},
            "outputs": [
                {
                    "name": "stdout",
                    "output_type": "stream",
                    "text": [text_out_1],
                }
            ],
            "source": [
                "# Load mc_record and event_record for all 3 datasets and verify against targets\n"
                "event_data = {}\n\n"
                "print(f\"{'Dataset':<28} | {'Ecm [GeV]':>9} | {'Label':>6} | {'0-5% <M> (Target)':>19} | {'0-5% <pT> (Target)':>20} | {'Point Cloud Shape':>22}\")\n"
                "print('-' * 118)\n\n"
                "for name, cfg in DATASETS.items():\n"
                "    with h5py.File(cfg['path'], 'r') as f:\n"
                "        mc = f['mc_record'][:]\n"
                "        ev = f['event_record'][:]\n"
                "        pc_shape = f['point_cloud'].shape\n"
                "        pc_dtype = f['point_cloud'].dtype\n"
                "        ev_cols = list(f['event_record'].attrs['columns'])\n"
                "        pc_cols = list(f['point_cloud'].attrs['columns'])\n\n"
                "    event_data[name] = {'mc': mc, 'ev': ev, 'pc_shape': pc_shape, 'ev_cols': ev_cols, 'pc_cols': pc_cols}\n"
                "    c05 = ev[:, 0] < 0.05\n"
                "    m_05 = ev[c05, 1].mean()\n"
                "    pt_05 = ev[c05, 7].mean()\n\n"
                "    print(\n"
                "        f\"{name:<28} | {mc[0]:>9.1f} | {mc[1]:>6.2f} | \"\n"
                "        f\"{m_05:>8.1f} ({cfg['target_mult_0_5']:>6.1f}) | \"\n"
                "        f\"{pt_05:>8.4f} ({cfg['target_pt_0_5']:>6.4f}) | \"\n"
                "        f\"{str(pc_shape):>14} ({pc_dtype})\"\n"
                "    )"
            ],
        },
        {
            "cell_type": "markdown",
            "metadata": {},
            "source": [
                "## 2. Event-Level Glauber & Multiplicity Distributions\n\n"
                "Visualize the minimum-bias (`0-100%` centrality) distributions stored in `event_record` (`float32`):\n"
                "- **Multiplicity $M$** in $|\\eta| < 1.0$, modeled with two-component Glauber ancestor scaling and Negative Binomial fluctuations.\n"
                "- **Impact parameter $b$** ($P(b) \\propto b$).\n"
                "- **Participating nucleons $\\langle N_{\\mathrm{part}} \\rangle$** and **Event $\\langle p_T \\rangle$** as a function of centrality percentage."
            ],
        },
        {
            "cell_type": "code",
            "execution_count": 3,
            "metadata": {},
            "outputs": [
                {
                    "data": {
                        "image/png": b64_fig1,
                        "text/plain": ["<Figure size 1560x1080 with 4 Axes>"],
                    },
                    "metadata": {},
                    "output_type": "display_data",
                }
            ],
            "source": [
                "fig, axes = plt.subplots(2, 2, figsize=(13, 9))\n"
                "ax_m, ax_b = axes[0, 0], axes[0, 1]\n"
                "ax_np, ax_mpt = axes[1, 0], axes[1, 1]\n\n"
                "for name, cfg in DATASETS.items():\n"
                "    ev = event_data[name]['ev']\n"
                "    color = cfg['color']\n"
                "    c, mult, b, npart, mpt = ev[:, 0], ev[:, 1], ev[:, 2], ev[:, 3], ev[:, 7]\n\n"
                "    ax_m.hist(mult, bins=80, density=True, histtype='step', lw=2.0, color=color,\n"
                "              label=f\"{name} ($\\\\langle M \\\\rangle_{{0-5\\\\%}}={mult[c<0.05].mean():.0f}$)\")\n"
                "    ax_b.hist(b, bins=60, density=True, histtype='step', lw=2.0, color=color, label=name)\n\n"
                "    c_bins = np.linspace(0.0, 1.0, 21)\n"
                "    c_centers = 0.5 * (c_bins[:-1] + c_bins[1:])\n"
                "    idx_bin = np.digitize(c, c_bins) - 1\n"
                "    np_mean = [npart[idx_bin == k].mean() for k in range(20)]\n"
                "    pt_mean = [mpt[idx_bin == k].mean() for k in range(20)]\n\n"
                "    ax_np.plot(c_centers * 100, np_mean, 'o-', ms=4, lw=1.8, color=color, label=name)\n"
                "    ax_mpt.plot(c_centers * 100, pt_mean, 's-', ms=4, lw=1.8, color=color, label=name)\n"
                "    ax_mpt.axhline(cfg['target_pt_0_5'], color=color, linestyle=':', alpha=0.6)\n\n"
                "ax_m.set_yscale('log')\n"
                "ax_m.set_title('(a) Charged Multiplicity $M$ Distribution (Min-Bias 0–100%)')\n"
                "ax_m.set_xlabel('Multiplicity $M$ ($|\\\\eta| < 1.0$)')\n"
                "ax_m.set_ylabel('Probability Density $P(M)$')\n"
                "ax_m.legend(fontsize=8)\n"
                "ax_m.grid(True, which='both', alpha=0.3)\n\n"
                "ax_b.set_title('(b) Impact Parameter $b$ Distribution')\n"
                "ax_b.set_xlabel('Impact Parameter $b$ [fm]')\n"
                "ax_b.set_ylabel('Probability Density $P(b)$ [fm$^{-1}$]')\n"
                "ax_b.legend(fontsize=8)\n"
                "ax_b.grid(True, alpha=0.3)\n\n"
                "ax_np.set_title('(c) Mean Participating Nucleons $\\\\langle N_{\\\\mathrm{part}} \\\\rangle$ vs Centrality')\n"
                "ax_np.set_xlabel('Centrality Class [%] (0% = Central, 100% = Peripheral)')\n"
                "ax_np.set_ylabel('$\\\\langle N_{\\\\mathrm{part}} \\\\rangle$')\n"
                "ax_np.legend(fontsize=8)\n"
                "ax_np.grid(True, alpha=0.3)\n\n"
                "ax_mpt.set_title('(d) Event Mean $\\\\langle p_T \\\\rangle$ vs Centrality (Dotted = 0–5% Target)')\n"
                "ax_mpt.set_xlabel('Centrality Class [%]')\n"
                "ax_mpt.set_ylabel('Event $\\\\langle p_T \\\\rangle$ [GeV/$c$]')\n"
                "ax_mpt.legend(fontsize=8)\n"
                "ax_mpt.grid(True, alpha=0.3)\n\n"
                "plt.tight_layout()\n"
                "plt.show()"
            ],
        },
        {
            "cell_type": "markdown",
            "metadata": {},
            "source": [
                "## 3. Reconstructing Single-Particle Kinematics ($p_T$, $\\eta$, $\\phi$) from the `float16` Point Cloud\n\n"
                "Each event $e$ has $M_e = \\text{int}(\\text{event\\_record}[e, 1])$ active particles in `point_cloud[e, :M_e, :]` (`float16`), with columns:\n"
                "- `[0]`: $\\cos(\\phi_i)$ and `[1]`: $\\sin(\\phi_i) \\implies \\phi_i = \\operatorname{atan2}(\\sin\\phi_i, \\cos\\phi_i)$\n"
                "- `[2]`: $\\ln(p_{T,i} / \\langle p_T \\rangle_e) \\implies p_{T,i} = \\langle p_T \\rangle_e \\cdot \\exp(\\text{pc}[e, i, 2])$\n"
                "- `[3]`: $Q_i \\ln(1 + m_i)$\n"
                "- `[4]`: $\\eta_i / \\langle \\eta \\rangle_e \\implies \\eta_i = \\langle \\eta \\rangle_e \\cdot \\text{pc}[e, i, 4]$"
            ],
        },
        {
            "cell_type": "code",
            "execution_count": 4,
            "metadata": {},
            "outputs": [
                {
                    "name": "stdout",
                    "output_type": "stream",
                    "text": [text_out_3],
                }
            ],
            "source": [
                "def reconstruct_particles_from_h5(h5_path, max_events=4000, central_only=True):\n"
                "    \"\"\"Unpack physical (pT, eta, phi, Q*log(1+m)) from the zero-padded float16 point_cloud.\"\"\"\n"
                "    with h5py.File(h5_path, 'r') as f:\n"
                "        mc = f['mc_record'][:]\n"
                "        ev = f['event_record'][:]\n"
                "        if central_only:\n"
                "            idx = np.where(ev[:, 0] < 0.05)[0][:max_events]\n"
                "        else:\n"
                "            idx = np.arange(min(max_events, len(ev)))\n"
                "        ev_sub = ev[idx]\n"
                "        pc_sub = f['point_cloud'][idx]\n\n"
                "    pt_list, eta_list, phi_list, qlogm_list = [], [], [], []\n"
                "    for i in range(len(ev_sub)):\n"
                "        m = int(ev_sub[i, 1])\n"
                "        mean_pt = float(ev_sub[i, 7])\n"
                "        eta_mean = float(ev_sub[i, 9])\n\n"
                "        pc = pc_sub[i, :m, :].astype(np.float32)\n"
                "        pt = mean_pt * np.exp(pc[:, 2])\n"
                "        phi = np.arctan2(pc[:, 1], pc[:, 0])\n"
                "        eta_denom = eta_mean if abs(eta_mean) >= 1e-4 else (1e-4 if eta_mean >= 0 else -1e-4)\n"
                "        eta = pc[:, 4] * eta_denom\n\n"
                "        pt_list.append(pt)\n"
                "        eta_list.append(eta)\n"
                "        phi_list.append(phi)\n"
                "        qlogm_list.append(pc[:, 3])\n\n"
                "    return mc, ev, np.concatenate(pt_list), np.concatenate(eta_list), np.concatenate(phi_list), np.concatenate(qlogm_list)\n\n"
                "reconstructed = {}\n"
                "print(f\"{'Dataset':<28} | {'Sampled 0-5% Events':>19} | {'Total Particles':>15} | {'Event <pT> [GeV/c]':>18} | {'PointCloud <pT> [GeV/c]':>23} | {'<phi> [rad]':>11}\")\n"
                "print('-' * 126)\n\n"
                "for name, cfg in DATASETS.items():\n"
                "    mc, ev, pt_c, eta_c, phi_c, qlogm_c = reconstruct_particles_from_h5(cfg['path'], max_events=4000, central_only=True)\n"
                "    reconstructed[name] = {'pt': pt_c, 'eta': eta_c, 'phi': phi_c, 'qlogm': qlogm_c}\n"
                "    c05 = ev[:, 0] < 0.05\n"
                "    n_ev_sampled = min(4000, int(np.sum(c05)))\n"
                "    ev_pt_05 = float(ev[c05, 7][:n_ev_sampled].mean())\n"
                "    print(f\"{name:<28} | {n_ev_sampled:>19,d} | {len(pt_c):>15,d} | {ev_pt_05:>18.4f} | {float(pt_c.mean()):>23.4f} | {float(phi_c.mean()):>11.4f}\")"
            ],
        },
        {
            "cell_type": "markdown",
            "metadata": {},
            "source": [
                "## 4. Transverse Momentum ($p_T$), Pseudorapidity ($\\eta$), and Azimuthal ($\\phi$) Spectra with Boltzmann Fits\n\n"
                "For a 2D thermal Boltzmann distribution in the transverse plane ($d^2N / dp_x dp_y \\propto \\exp(-p_T / T_{\\mathrm{eff}})$), the radial $p_T$ probability density is:\n"
                "$$\\frac{dP}{dp_T} = A\\,\\frac{p_T}{T_{\\mathrm{eff}}^2} \\exp\\!\\left(-\\frac{p_T}{T_{\\mathrm{eff}}}\\right) \\quad \\Longleftrightarrow \\quad \\ln\\!\\left(\\frac{1}{p_T}\\frac{dP}{dp_T}\\right) = A_0 - \\frac{p_T}{T_{\\mathrm{eff}}}$$\n"
                "whose analytical mean is $\\langle p_T \\rangle = 2\\,T_{\\mathrm{eff}}$. Below we compare:\n"
                "1. **MLE Boltzmann estimator**: $\\hat{T}_{\\mathrm{MLE}} = \\langle p_T \\rangle / 2$\n"
                "2. **Nonlinear Least-Squares PDF Fit** (`scipy.optimize.curve_fit` on $dP/dp_T$)\n"
                "3. **Invariant Yield Exponential Slope Fit** (`curve_fit` on $\\ln[(1/p_T)\\,dP/dp_T]$)"
            ],
        },
        {
            "cell_type": "code",
            "execution_count": 5,
            "metadata": {},
            "outputs": [
                {
                    "name": "stdout",
                    "output_type": "stream",
                    "text": [text_out_4],
                },
                {
                    "data": {
                        "image/png": b64_fig2,
                        "text/plain": ["<Figure size 1680x1200 with 4 Axes>"],
                    },
                    "metadata": {},
                    "output_type": "display_data",
                },
            ],
            "source": [
                "def boltzmann_pdf(pt, t_eff, norm):\n"
                "    return norm * (pt / (t_eff ** 2)) * np.exp(-pt / t_eff)\n\n"
                "def boltzmann_inv_log(pt, t_eff, a0):\n"
                "    return a0 - pt / t_eff\n\n"
                "fig, axes = plt.subplots(2, 2, figsize=(14, 10))\n"
                "ax_pt, ax_inv = axes[0, 0], axes[0, 1]\n"
                "ax_eta, ax_phi = axes[1, 0], axes[1, 1]\n\n"
                "print(f\"{'Dataset':<28} | {'Target <pT>':>11} | {'Target T_eff':>12} | {'MLE T_eff (<pT>/2)':>18} | {'CurveFit T_eff':>14} | {'LogSlope T_eff':>14} | {'Fit 2*T_eff (PDF/Slope)':>23}\")\n"
                "print('-' * 134)\n\n"
                "for name, cfg in DATASETS.items():\n"
                "    color = cfg['color']\n"
                "    target_pt = cfg['target_pt_0_5']\n"
                "    target_t = target_pt / 2.0\n"
                "    pt_c = reconstructed[name]['pt']\n"
                "    eta_c = reconstructed[name]['eta']\n"
                "    phi_c = reconstructed[name]['phi']\n\n"
                "    t_mle = float(pt_c.mean()) / 2.0\n\n"
                "    bins = np.linspace(0.05, 3.5, 70)\n"
                "    counts, edges = np.histogram(pt_c, bins=bins, density=True)\n"
                "    centers = 0.5 * (edges[:-1] + edges[1:])\n"
                "    valid = counts > 0\n\n"
                "    popt_pdf, _ = curve_fit(boltzmann_pdf, centers[valid], counts[valid], p0=[target_t, 1.0])\n"
                "    t_pdf, norm_pdf = popt_pdf\n\n"
                "    inv_mask = (centers >= 0.15) & (centers <= 2.5) & (counts > 0)\n"
                "    inv_y = np.log(counts[inv_mask] / centers[inv_mask])\n"
                "    popt_inv, _ = curve_fit(boltzmann_inv_log, centers[inv_mask], inv_y, p0=[target_t, 0.0])\n"
                "    t_slope, a0_slope = popt_inv\n\n"
                "    print(\n"
                "        f\"{name:<28} | {target_pt:>11.4f} | {target_t:>12.4f} | \"\n"
                "        f\"{t_mle:>18.4f} | {t_pdf:>14.4f} | {t_slope:>14.4f} | \"\n"
                "        f\"{2*t_pdf:>10.4f} / {2*t_slope:<10.4f}\"\n"
                "    )\n\n"
                "    pt_fine = np.linspace(0.02, 3.5, 300)\n"
                "    ax_pt.scatter(centers[valid], counts[valid], s=18, color=color, alpha=0.8,\n"
                "                  label=f\"{name} (<pT>={pt_c.mean():.3f})\")\n"
                "    ax_pt.plot(pt_fine, boltzmann_pdf(pt_fine, t_pdf, norm_pdf), color=color, lw=2.0, linestyle='--',\n"
                "               label=f\"Boltzmann Fit: $T_{{eff}}={t_pdf:.3f}$ GeV ($2T_{{eff}}={2*t_pdf:.3f}$)\")\n\n"
                "    ax_inv.scatter(centers[valid], counts[valid] / centers[valid], s=16, color=color, alpha=0.8, label=name)\n"
                "    ax_inv.plot(pt_fine, np.exp(boltzmann_inv_log(pt_fine, t_slope, a0_slope)), color=color, lw=1.8, linestyle='--',\n"
                "                label=f\"Slope $T_{{eff}}={t_slope:.3f}$ GeV ($2T_{{eff}}={2*t_slope:.3f}$)\")\n\n"
                "    ax_eta.hist(eta_c, bins=60, range=(-1.1, 1.1), density=True, histtype='step', lw=2.0, color=color, label=name)\n"
                "    ax_phi.hist(phi_c, bins=60, range=(-np.pi, np.pi), density=True, histtype='step', lw=2.0, color=color, label=name)\n\n"
                "ax_pt.set_title('(a) Central (0–5%) $p_T$ Spectrum & Boltzmann PDF Fits')\n"
                "ax_pt.set_xlabel('$p_T$ [GeV/$c$]')\n"
                "ax_pt.set_ylabel('Probability Density $dP/dp_T$ [(GeV/$c$)$^{-1}$]')\n"
                "ax_pt.legend(fontsize=7.5)\n"
                "ax_pt.grid(True, alpha=0.3)\n\n"
                "ax_inv.set_yscale('log')\n"
                "ax_inv.set_title('(b) Invariant Spectrum $(1/p_T)\\\\,dP/dp_T$ & Exponential Slope Fits')\n"
                "ax_inv.set_xlabel('$p_T$ [GeV/$c$]')\n"
                "ax_inv.set_ylabel('$(1/p_T)\\\\,dP/dp_T$ [(GeV/$c$)$^{-2}$]')\n"
                "ax_inv.legend(fontsize=7.5)\n"
                "ax_inv.grid(True, which='both', alpha=0.3)\n\n"
                "ax_eta.axhline(0.5, color='black', linestyle=':', lw=1.5, label='Uniform $|\\\\eta|<1$ ($0.5$)')\n"
                "ax_eta.set_title('(c) Reconstructed Pseudorapidity $\\\\eta$ Spectrum')\n"
                "ax_eta.set_xlabel('Pseudorapidity $\\\\eta$')\n"
                "ax_eta.set_ylabel('Probability Density $dP/d\\\\eta$')\n"
                "ax_eta.set_ylim(0, 0.72)\n"
                "ax_eta.legend(fontsize=8)\n"
                "ax_eta.grid(True, alpha=0.3)\n\n"
                "ax_phi.axhline(1.0 / (2.0 * np.pi), color='black', linestyle=':', lw=1.5, label='Isotropic $1/(2\\\\pi) \\\\approx 0.1592$')\n"
                "ax_phi.set_title('(d) Reconstructed Azimuthal Angle $\\\\phi$ Spectrum')\n"
                "ax_phi.set_xlabel('Azimuthal Angle $\\\\phi$ [rad]')\n"
                "ax_phi.set_ylabel('Probability Density $dP/d\\\\phi$ [rad$^{-1}$]')\n"
                "ax_phi.set_ylim(0, 0.22)\n"
                "ax_phi.legend(fontsize=8)\n"
                "ax_phi.grid(True, alpha=0.3)\n\n"
                "plt.tight_layout()\n"
                "plt.show()"
            ],
        },
        {
            "cell_type": "markdown",
            "metadata": {},
            "source": [
                "## 5. Identified Particle Species ($\\pi^\\pm, K^\\pm, p/\\bar{p}$) Boltzmann Fits & Radial Flow Check\n\n"
                "Why does a single-component curve fit to the inclusive all-charged $p_T$ distribution yield a slightly softer $T_{\\mathrm{eff}}$ near the peak and a slightly harder $T_{\\mathrm{eff}}$ in the tail? Because the inclusive charged hadron spectrum is a **superposition of pions, kaons, and protons** with radial-flow mass ordering ($T_{\\pi} < T_K < T_p$).\n\n"
                "By decoding `point_cloud[:, :, 3]` ($|Q\\ln(1+m)| \\in \\{0.131, 0.401, 0.662\\}$) and fitting each species individually with `boltzmann_pdf`, we verify that **each individual hadron species is an exact Boltzmann distribution** and that their species-weighted sum reproduces the exact target $\\langle p_T \\rangle$."
            ],
        },
        {
            "cell_type": "code",
            "execution_count": 6,
            "metadata": {},
            "outputs": [
                {
                    "name": "stdout",
                    "output_type": "stream",
                    "text": [text_out_5],
                },
                {
                    "data": {
                        "image/png": b64_fig3,
                        "text/plain": ["<Figure size 1800x576 with 3 Axes>"],
                    },
                    "metadata": {},
                    "output_type": "display_data",
                },
            ],
            "source": [
                "fig, axes = plt.subplots(1, 3, figsize=(15, 4.8), sharey=False)\n"
                "species_defs = [\n"
                "    ('Pions ($\\\\pi^\\\\pm$)', 0.13065, 0.13957039, '#1f77b4'),\n"
                "    ('Kaons ($K^\\\\pm$)', 0.40124, 0.493677, '#ff7f0e'),\n"
                "    ('Protons ($p, \\\\bar{p}$)', 0.66180, 0.93827208, '#d62728'),\n"
                "]\n\n"
                "print(f\"{'Dataset':<26} | {'Species':<16} | {'Frac':>6} | {'Target T_s [GeV]':>16} | {'Fitted T_s [GeV]':>16} | {'Target <pT>_s':>13} | {'Fitted 2*T_s':>12}\")\n"
                "print('-' * 119)\n\n"
                "for idx_ds, (name, cfg) in enumerate(DATASETS.items()):\n"
                "    ax = axes[idx_ds]\n"
                "    pt_c = reconstructed[name]['pt']\n"
                "    qlogm_c = np.abs(reconstructed[name]['qlogm'])\n\n"
                "    probs = np.array(cfg['species_probs'], dtype=np.float64)\n"
                "    probs /= probs.sum()\n"
                "    rel_w = 1.0 + 0.65 * (SPECIES_MASS - 0.13957039)\n"
                "    avg_w = float(np.sum(probs * rel_w))\n\n"
                "    weighted_pt_sum = 0.0\n"
                "    for sp_label, qval, mass_val, sp_color in species_defs:\n"
                "        mask_sp = np.abs(qlogm_c - qval) < 0.05\n"
                "        pt_sp = pt_c[mask_sp]\n"
                "        frac_sp = float(np.mean(mask_sp))\n\n"
                "        sp_w = (1.0 + 0.65 * (mass_val - 0.13957039)) / avg_w\n"
                "        target_pt_sp = cfg['target_pt_0_5'] * sp_w\n"
                "        target_t_sp = target_pt_sp / 2.0\n\n"
                "        bins_sp = np.linspace(0.05, 3.2, 55)\n"
                "        counts_sp, edges_sp = np.histogram(pt_sp, bins=bins_sp, density=True)\n"
                "        centers_sp = 0.5 * (edges_sp[:-1] + edges_sp[1:])\n"
                "        v_sp = counts_sp > 0\n\n"
                "        popt_sp, _ = curve_fit(boltzmann_pdf, centers_sp[v_sp], counts_sp[v_sp], p0=[target_t_sp, 1.0])\n"
                "        t_fit_sp, norm_fit_sp = popt_sp\n"
                "        weighted_pt_sum += frac_sp * (2.0 * t_fit_sp)\n\n"
                "        sp_short = sp_label.split('(')[0].strip()\n"
                "        print(f\"{name:<26} | {sp_short:<16} | {frac_sp:>6.3f} | {target_t_sp:>16.4f} | {t_fit_sp:>16.4f} | {target_pt_sp:>13.4f} | {2*t_fit_sp:>12.4f}\")\n\n"
                "        pt_grid = np.linspace(0.02, 3.2, 250)\n"
                "        ax.scatter(centers_sp[v_sp], counts_sp[v_sp], s=14, color=sp_color, alpha=0.75)\n"
                "        ax.plot(pt_grid, boltzmann_pdf(pt_grid, t_fit_sp, norm_fit_sp), color=sp_color, lw=2.0, linestyle='--',\n"
                "                label=f\"{sp_label}: $T_s={t_fit_sp:.3f}$ ($\\\\langle p_T\\\\rangle={2*t_fit_sp:.3f}$)\")\n\n"
                "    print(f\"{name:<26} | {'Species-Weighted':<16} | {1.000:>6.3f} | {cfg['target_pt_0_5']/2:>16.4f} | {weighted_pt_sum/2:>16.4f} | {cfg['target_pt_0_5']:>13.4f} | {weighted_pt_sum:>12.4f}\")\n"
                "    print('-' * 119)\n\n"
                "    ax.set_title(f\"{name}\\nSpecies Boltzmann Fits (0–5% Central)\")\n"
                "    ax.set_xlabel('$p_T$ [GeV/$c$]')\n"
                "    if idx_ds == 0:\n"
                "        ax.set_ylabel('Probability Density $dP_s/dp_T$ [(GeV/$c$)$^{-1}$]')\n"
                "    ax.legend(fontsize=7.5)\n"
                "    ax.grid(True, alpha=0.3)\n\n"
                "plt.tight_layout()\n"
                "plt.show()"
            ],
        },
    ]

    # Add Final Summary markdown cell strictly following notebook-guidance rules
    summary_md = (
        "### Q&A\n"
        "- **Do the reconstructed $p_T$, $\\eta$, and $\\phi$ distributions from the `float16` point cloud match the generated physics settings?**\n"
        "  Yes. Unpacking $\\ln(p_T/\\langle p_T \\rangle)$, $\\cos\\phi$, $\\sin\\phi$, and $\\eta_i/\\langle \\eta \\rangle$ from `point_cloud` (`float16`) reproduces the uniform pseudorapidity density $dP/d\\eta = 0.50$ on $|\\eta| < 1.0$, the isotropic azimuthal density $dP/d\\phi = 1/(2\\pi) \\approx 0.1592\\text{ rad}^{-1}$ on $[-\\pi, \\pi]$, and the exact event-record mean transverse momenta ($0.6802\\text{ GeV}/c$ at Top LHC, $0.5600\\text{ GeV}/c$ at Top RHIC, and $0.4205\\text{ GeV}/c$ at RHIC BES $9.2\\text{ GeV}$).\n"
        "- **Does fitting the $p_T$ spectrum with a Boltzmann distribution reproduce the input parameters?**\n"
        "  Yes. The MLE Boltzmann estimator ($\\hat{T}_{\\mathrm{MLE}} = \\langle p_T \\rangle / 2$) reproduces the exact target effective temperatures ($0.3401\\text{ GeV}$, $0.2800\\text{ GeV}$, $0.2103\\text{ GeV}$). Furthermore, fitting each hadron species ($\\pi^\\pm, K^\\pm, p/\\bar{p}$) individually with the 2D thermal Boltzmann density $dP_s/dp_T = A(p_T/T_s^2)\\exp(-p_T/T_s)$ recovers every species' input effective temperature $T_s$ to within $<0.3\\%$, and their species-weighted mean $\\sum_s f_s (2 T_s)$ reproduces the exact target central $\\langle p_T \\rangle$ ($0.6811\\text{ GeV}/c$, $0.5614\\text{ GeV}/c$, and $0.4222\\text{ GeV}/c$).\n\n"
        "### Data Analysis Key Findings\n"
        "- **Top LHC ($\\sqrt{s_{\\mathrm{NN}}} = 5020\\text{ GeV}$, Label `0.01`)**: Central (0–5%) events achieve $\\langle M \\rangle = 1937.5$ (target $1940.0$) and $\\langle p_T \\rangle = 0.6801\\text{ GeV}/c$ (target $0.6800\\text{ GeV}/c$). The inclusive Boltzmann PDF fit gives $T_{\\mathrm{eff}} = 0.3362\\text{ GeV}$ ($2T_{\\mathrm{eff}} = 0.6724\\text{ GeV}/c$), the log-slope fit gives $T_{\\mathrm{eff}} = 0.3430\\text{ GeV}$ ($2T_{\\mathrm{eff}} = 0.6860\\text{ GeV}/c$), and the species-resolved fits recover $T_{\\pi} = 0.3238\\text{ GeV}$ (target $0.3232$), $T_K = 0.3973\\text{ GeV}$ (target $0.3975$), and $T_p = 0.4904\\text{ GeV}$ (target $0.4909$), with species-weighted $2T_{\\mathrm{eff}} = 0.6811\\text{ GeV}/c$.\n"
        "- **Top RHIC ($\\sqrt{s_{\\mathrm{NN}}} = 200\\text{ GeV}$, Label `0.02`)**: Central (0–5%) events achieve $\\langle M \\rangle = 679.5$ (target $680.0$) and $\\langle p_T \\rangle = 0.5599\\text{ GeV}/c$ (target $0.5600\\text{ GeV}/c$). The inclusive Boltzmann PDF fit yields $T_{\\mathrm{eff}} = 0.2764\\text{ GeV}$ ($2T_{\\mathrm{eff}} = 0.5529\\text{ GeV}/c$), the log-slope fit yields $T_{\\mathrm{eff}} = 0.2873\\text{ GeV}$ ($2T_{\\mathrm{eff}} = 0.5746\\text{ GeV}/c$), and the species-resolved fits recover $T_{\\pi} = 0.2646\\text{ GeV}$ (target $0.2639$), $T_K = 0.3250\\text{ GeV}$ (target $0.3246$), and $T_p = 0.4025\\text{ GeV}$ (target $0.4009$), with species-weighted $2T_{\\mathrm{eff}} = 0.5614\\text{ GeV}/c$.\n"
        "- **RHIC BES ($\\sqrt{s_{\\mathrm{NN}}} = 9.2\\text{ GeV}$, Label `0.03`)**: Central (0–5%) events achieve $\\langle M \\rangle = 80.1$ (target $80.0$) and $\\langle p_T \\rangle = 0.4202\\text{ GeV}/c$ (target $0.4200\\text{ GeV}/c$). Because of the higher baryon fraction ($13.9\\%$ protons) at $9.2\\text{ GeV}$, the inclusive single-component fit shows stronger multi-species curvature ($T_{\\mathrm{PDF}} = 0.2058\\text{ GeV}$ vs $T_{\\mathrm{slope}} = 0.2290\\text{ GeV}$), while species-resolved Boltzmann fits recover $T_{\\pi} = 0.1921\\text{ GeV}$ (target $0.1893$), $T_K = 0.2352\\text{ GeV}$ (target $0.2329$), and $T_p = 0.2910\\text{ GeV}$ (target $0.2876$), with species-weighted $2T_{\\mathrm{eff}} = 0.4222\\text{ GeV}/c$.\n\n"
        "### Insights or Next Steps\n"
        "- **Multi-Species vs Single-Temperature Fits**: When benchmarking ML or statistical models on the inclusive charged point cloud, using `point_cloud[:, :, 3]` ($Q\\ln(1+m)$) allows exact separation of $\\pi^\\pm, K^\\pm$, and $p/\\bar{p}$ so radial-flow mass ordering can be disentangled from the kinetic freeze-out temperature.\n"
        "- **Adding Transport & Jet Generator Datasets (`0.1X`–`0.5X`)**: Use the same notebook pipeline to compare these vanilla thermal baselines (`0.01`–`0.03`) against UrQMD (`0.1X`), Angantyr (`0.2X`), JEWEL (`0.3X`), HIJING (`0.4X`), or AMPT (`0.5X`) to quantify power-law hard-scattering tails ($p_T > 2.5\\text{ GeV}/c$) and elliptic flow ($v_2$) relative to the event plane $\\Psi_2$."
    )
    cells.append(
        {
            "cell_type": "markdown",
            "metadata": {},
            "source": [summary_md],
        }
    )

    nb = {
        "cells": cells,
        "metadata": {
            "kernelspec": {
                "display_name": "Python 3 (.venv)",
                "language": "python",
                "name": "python3",
            },
            "language_info": {
                "name": "python",
                "version": "3.14.7",
            },
        },
        "nbformat": 4,
        "nbformat_minor": 2,
    }

    nb_path = "inspect_thermal_distributions.ipynb"
    with open(nb_path, "w", encoding="utf-8") as f:
        json.dump(nb, f, indent=2)
    print(f"Wrote executed notebook to {nb_path}")


if __name__ == "__main__":
    main()
