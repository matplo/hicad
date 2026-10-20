#!/usr/bin/env python3
"""
Local HTTP streaming server for HDF5 heavy-ion point-cloud event files.

Serves:
  - GET /                       -> h5_event_display.html (Three.js WebGPU 3D display)
  - GET /api/files              -> List of discovered .h5 files with mc_record metadata
  - GET /api/event?...          -> Single event record + active (non-padded) point_cloud particles

Usage:
  henv -x python serve_h5_events.py [--port 8765] [--data-dir output]
  # or
  .venv/bin/python serve_h5_events.py [--port 8765] [--data-dir output]
"""

from __future__ import annotations

import argparse
import glob
import json
import os
import sys
import threading
import webbrowser
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from typing import Any, Dict, List, Optional
from urllib.parse import parse_qs, urlparse

import h5py
import numpy as np


# Species lookup from |Q * log(1 + mass)| in GeV/c^2:
# pi+- (m=0.13957): ln(1+m) = 0.13065
# K+-  (m=0.49368): ln(1+m) = 0.40124
# p/pbar (m=0.93827): ln(1+m) = 0.66180
SPECIES_ABS_QLOGM = np.array([0.13065, 0.40124, 0.66180], dtype=np.float32)
SPECIES_MASSES = np.array([0.13957039, 0.493677, 0.938272088], dtype=np.float32)


def decode_mc_label(label: float, ecm: float) -> Dict[str, str]:
    """Decode MC model family and collision system tag from mc_record [Ecm, Label]."""
    family_code = int(np.floor(round(label, 4) * 10.0 + 1e-5))
    families = {
        0: "Vanilla Thermal (Rita)",
        1: "UrQMD",
        2: "Angantyr",
        3: "JEWEL",
        4: "HIJING",
        5: "AMPT",
    }
    model_name = families.get(family_code, f"MC Model ({label:.2f})")

    if ecm >= 1000.0:
        sys_name = f"Pb–Pb · √s_NN = {ecm / 1000.0:.2f} TeV"
    elif ecm >= 100.0:
        sys_name = f"Au–Au · √s_NN = {ecm:.0f} GeV"
    else:
        sys_name = f"Au–Au BES · √s_NN = {ecm:.1f} GeV"

    return {
        "model_name": model_name,
        "system_tag": sys_name,
    }


class H5DatasetCatalog:
    """Thread-safe catalog of HDF5 event files with in-memory event_record index and lazy point_cloud slicing."""

    def __init__(self, search_dirs: List[str]) -> None:
        self.lock = threading.Lock()
        self.files: Dict[str, Dict[str, Any]] = {}
        self.rng = np.random.default_rng(42)
        self._discover(search_dirs)

    def _discover(self, search_dirs: List[str]) -> None:
        candidates: List[str] = []
        for d in search_dirs:
            if not d:
                continue
            d_exp = os.path.abspath(os.path.expanduser(d))
            if os.path.isfile(d_exp) and d_exp.endswith(".h5"):
                candidates.append(d_exp)
            elif os.path.isdir(d_exp):
                candidates.extend(sorted(glob.glob(os.path.join(d_exp, "*.h5"))))
                candidates.extend(sorted(glob.glob(os.path.join(d_exp, "output", "*.h5"))))

        seen_names = set()
        for path in candidates:
            fname = os.path.basename(path)
            if fname in seen_names or not os.path.isfile(path):
                continue
            try:
                with h5py.File(path, "r") as f:
                    if "mc_record" not in f or "event_record" not in f or "point_cloud" not in f:
                        continue
                    mc = f["mc_record"][:].astype(np.float32)
                    ev = f["event_record"][:].astype(np.float32)
                    pc_shape = f["point_cloud"].shape
                    pc_dtype = str(f["point_cloud"].dtype)

                ecm = float(mc[0])
                label = float(mc[1])
                meta = decode_mc_label(label, ecm)
                c_mask = ev[:, 0] < 0.05
                mean_mult_0_5 = float(ev[c_mask, 1].mean()) if np.any(c_mask) else float(ev[:, 1].mean())
                mean_pt_0_5 = float(ev[c_mask, 7].mean()) if np.any(c_mask) else float(ev[:, 7].mean())

                self.files[fname] = {
                    "id": fname,
                    "name": fname,
                    "_path": path,
                    "size_mb": round(os.path.getsize(path) / (1024 * 1024), 2),
                    "ecm": round(ecm, 4),
                    "label": round(label, 4),
                    "model_name": meta["model_name"],
                    "system_tag": meta["system_tag"],
                    "n_events": int(ev.shape[0]),
                    "max_m": int(pc_shape[1]),
                    "point_cloud_dtype": pc_dtype,
                    "mean_mult_0_5": round(mean_mult_0_5, 1),
                    "mean_pt_0_5": round(mean_pt_0_5, 4),
                    "mean_mult_mb": round(float(ev[:, 1].mean()), 1),
                    "mean_pt_mb": round(float(ev[:, 7].mean()), 4),
                    "_event_record": ev,  # cached (100k x 10 float32 = 4 MB) for instant filtering
                }
                seen_names.add(fname)
            except Exception as exc:
                print(f"[WARN] Skipping {path}: {exc}", file=sys.stderr)

        # Sort by Ecm descending (LHC 5020 -> RHIC 200 -> BES 9.2)
        sorted_items = sorted(self.files.values(), key=lambda x: -x["ecm"])
        self.files = {item["id"]: item for item in sorted_items}

    def list_files_public(self) -> List[Dict[str, Any]]:
        out = []
        for info in self.files.values():
            out.append({k: v for k, v in info.items() if not k.startswith("_")})
        return out

    def get_event(
        self,
        file_id: Optional[str] = None,
        idx: Optional[int] = None,
        nav: str = "exact",
        cur_idx: int = 0,
        c_min: float = 0.0,
        c_max: float = 1.0,
    ) -> Dict[str, Any]:
        if not self.files:
            raise ValueError("No HDF5 files loaded in catalog.")

        if not file_id or file_id not in self.files:
            file_id = next(iter(self.files.keys()))

        info = self.files[file_id]
        ev_table: np.ndarray = info["_event_record"]
        n_events = info["n_events"]

        # Select event index based on navigation mode and centrality filter [c_min, c_max]
        c_col = ev_table[:, 0]
        matching = np.flatnonzero((c_col >= c_min) & (c_col <= c_max))
        if matching.size == 0:
            matching = np.arange(n_events, dtype=np.int64)

        if idx is not None and nav == "exact":
            chosen_idx = int(np.clip(idx, 0, n_events - 1))
        elif nav == "next":
            greater = matching[matching > cur_idx]
            chosen_idx = int(greater[0]) if greater.size > 0 else int(matching[0])
        elif nav == "prev":
            lesser = matching[matching < cur_idx]
            chosen_idx = int(lesser[-1]) if lesser.size > 0 else int(matching[-1])
        else:
            # random within matching centrality range
            with self.lock:
                chosen_idx = int(self.rng.choice(matching))

        ev_row = ev_table[chosen_idx]
        centrality = float(ev_row[0])
        mult = int(round(float(ev_row[1])))
        b_fm = float(ev_row[2])
        n_part = float(ev_row[3])
        n_coll = float(ev_row[4])
        sin_psi2 = float(ev_row[5])
        cos_psi2 = float(ev_row[6])
        mean_pt = float(ev_row[7])
        sum_pt = float(ev_row[8])
        eta_mean = float(ev_row[9])

        psi2_rad = float(0.5 * np.arctan2(sin_psi2, cos_psi2))
        psi2_deg = float(np.degrees(psi2_rad))

        # Slice ONLY the active particles [0 : mult] for this single event from the HDF5 point_cloud
        with self.lock:
            with h5py.File(info["_path"], "r") as h5f:
                pc_slice = h5f["point_cloud"][chosen_idx, :mult, :].astype(np.float32)

        # Unpack point_cloud columns:
        # [0] cos(phi), [1] sin(phi), [2] log(pT/mean_pT), [3] Q*log(1+mass), [4] eta_i - eta_mean
        cos_phi = pc_slice[:, 0]
        sin_phi = pc_slice[:, 1]
        # Re-normalize unit circle slightly to remove float16 rounding drift
        norm_phi = np.hypot(cos_phi, sin_phi)
        norm_phi = np.where(norm_phi > 1e-6, norm_phi, 1.0)
        cos_phi = cos_phi / norm_phi
        sin_phi = sin_phi / norm_phi
        phi = np.arctan2(sin_phi, cos_phi)

        log_pt_ratio = pc_slice[:, 2]
        pt = mean_pt * np.exp(log_pt_ratio)

        q_log_m = pc_slice[:, 3]
        q = np.where(q_log_m >= 0.0, 1, -1).astype(np.int32)
        abs_qlogm = np.abs(q_log_m)

        # Nearest species match: 0 -> pi+-, 1 -> K+-, 2 -> p/pbar
        sp_idx = np.argmin(np.abs(abs_qlogm[:, None] - SPECIES_ABS_QLOGM[None, :]), axis=1).astype(np.int32)
        mass = SPECIES_MASSES[sp_idx]

        eta_diff = pc_slice[:, 4]
        eta = np.clip(eta_diff + eta_mean, -2.5, 2.5)

        n_pi = int(np.sum(sp_idx == 0))
        n_k = int(np.sum(sp_idx == 1))
        n_p = int(np.sum(sp_idx == 2))
        n_pos = int(np.sum(q > 0))
        n_neg = int(np.sum(q < 0))

        return {
            "file": info["id"],
            "mc_record": {
                "ecm": info["ecm"],
                "label": info["label"],
                "model_name": info["model_name"],
                "system_tag": info["system_tag"],
            },
            "event_idx": chosen_idx,
            "n_events": n_events,
            "matching_events": int(matching.size),
            "event_record": {
                "centrality": round(centrality, 5),
                "centrality_pct": round(centrality * 100.0, 2),
                "multiplicity": mult,
                "b": round(b_fm, 3),
                "n_part": round(n_part, 1),
                "n_coll": round(n_coll, 1),
                "sin_psi2": round(sin_psi2, 4),
                "cos_psi2": round(cos_psi2, 4),
                "psi2_rad": round(psi2_rad, 4),
                "psi2_deg": round(psi2_deg, 1),
                "mean_pt": round(mean_pt, 4),
                "sum_pt": round(sum_pt, 2),
                "eta_mean": round(eta_mean, 6),
            },
            "species_counts": {
                "pi": n_pi,
                "k": n_k,
                "p": n_p,
                "pos": n_pos,
                "neg": n_neg,
            },
            "particles": {
                "cos_phi": np.round(cos_phi, 4).tolist(),
                "sin_phi": np.round(sin_phi, 4).tolist(),
                "phi": np.round(phi, 4).tolist(),
                "pt": np.round(pt, 4).tolist(),
                "log_pt_ratio": np.round(log_pt_ratio, 4).tolist(),
                "q_log_m": np.round(q_log_m, 4).tolist(),
                "eta": np.round(eta, 4).tolist(),
                "eta_diff": np.round(eta_diff, 4).tolist(),
                "q": q.tolist(),
                "sp": sp_idx.tolist(),
                "mass": np.round(mass, 4).tolist(),
            },
        }


def make_handler(catalog: H5DatasetCatalog, html_path: str):
    class EventDisplayRequestHandler(BaseHTTPRequestHandler):
        def log_message(self, fmt: str, *args: Any) -> None:
            # Keep console logs concise
            sys.stdout.write(f"[HTTP] {self.address_string()} - {fmt % args}\n")

        def _send_json(self, payload: Any, status: int = HTTPStatus.OK) -> None:
            raw = json.dumps(payload).encode("utf-8")
            self.send_response(status)
            self.send_header("Content-Type", "application/json; charset=utf-8")
            self.send_header("Content-Length", str(len(raw)))
            self.send_header("Cache-Control", "no-store")
            self.send_header("Access-Control-Allow-Origin", "*")
            self.end_headers()
            self.wfile.write(raw)

        def _send_html(self, path: str) -> None:
            if not os.path.isfile(path):
                self.send_error(HTTPStatus.NOT_FOUND, f"HTML file not found: {path}")
                return
            with open(path, "rb") as f:
                raw = f.read()
            self.send_response(HTTPStatus.OK)
            self.send_header("Content-Type", "text/html; charset=utf-8")
            self.send_header("Content-Length", str(len(raw)))
            self.send_header("Cache-Control", "no-store, no-cache, must-revalidate, max-age=0")
            self.send_header("Pragma", "no-cache")
            self.send_header("Expires", "0")
            self.end_headers()
            self.wfile.write(raw)

        def do_GET(self) -> None:
            parsed = urlparse(self.path)
            route = parsed.path
            qs = parse_qs(parsed.query)

            if route in ("/", "/index.html", "/h5_event_display.html"):
                self._send_html(html_path)
                return

            if route == "/api/files":
                self._send_json({"files": catalog.list_files_public()})
                return

            if route == "/api/event":
                try:
                    file_id = qs.get("file", [None])[0]
                    nav = qs.get("nav", ["random"])[0]
                    idx_str = qs.get("idx", [None])[0]
                    cur_str = qs.get("cur", ["0"])[0]
                    c_min_str = qs.get("c_min", ["0.0"])[0]
                    c_max_str = qs.get("c_max", ["1.0"])[0]

                    idx = int(idx_str) if idx_str is not None and idx_str != "" else None
                    cur_idx = int(cur_str) if cur_str else 0
                    c_min = float(c_min_str)
                    c_max = float(c_max_str)

                    data = catalog.get_event(
                        file_id=file_id,
                        idx=idx,
                        nav=nav,
                        cur_idx=cur_idx,
                        c_min=c_min,
                        c_max=c_max,
                    )
                    self._send_json(data)
                except Exception as exc:
                    self._send_json({"error": str(exc)}, status=HTTPStatus.BAD_REQUEST)
                return

            self.send_error(HTTPStatus.NOT_FOUND, f"Unknown endpoint: {route}")

    return EventDisplayRequestHandler


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Stream HDF5 heavy-ion point-cloud events to a 3D Three.js WebGPU event display."
    )
    parser.add_argument(
        "--data-dir",
        type=str,
        default="output",
        help="Directory (or .h5 file path) containing the generated .h5 files (default: output).",
    )
    parser.add_argument(
        "--html",
        type=str,
        default="",
        help="Path to h5_event_display.html (defaults to h5_event_display.html next to this script).",
    )
    parser.add_argument(
        "--host",
        type=str,
        default="127.0.0.1",
        help="Host interface to bind (default: 127.0.0.1).",
    )
    parser.add_argument(
        "--port",
        type=int,
        default=8765,
        help="Port to serve on (default: 8765).",
    )
    parser.add_argument(
        "--open-browser",
        action="store_true",
        help="Automatically open the event display URL in the default browser.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    script_dir = os.path.dirname(os.path.abspath(__file__))
    html_path = (
        os.path.abspath(args.html)
        if args.html
        else os.path.join(script_dir, "h5_event_display.html")
    )

    search_dirs = [
        args.data_dir,
        os.path.join(script_dir, args.data_dir),
        os.path.join(script_dir, "output"),
        os.path.join(script_dir, "..", "boltzmann", "output"),
        os.path.join(script_dir, "..", "boltzmann"),
        os.path.join(script_dir, "boltzmann", "output"),
        script_dir,
    ]
    catalog = H5DatasetCatalog(search_dirs)
    files = catalog.list_files_public()

    print("=" * 78)
    print("HICAD HDF5 3D Point-Cloud Event Display Server")
    print("=" * 78)
    if not files:
        print("[WARN] No .h5 files found! Run `generate_thermal_h5.py` first or pass `--data-dir <path>`.")
    else:
        print(f"Loaded {len(files)} HDF5 dataset(s):")
        for f in files:
            print(
                f"  - {f['name']:32s} | Ecm={f['ecm']:6.1f} GeV | Label={f['label']:.2f} "
                f"({f['model_name']}) | N={f['n_events']:,} | max_M={f['max_m']}"
            )

    url = f"http://{args.host}:{args.port}/"
    print("-" * 78)
    print(f"Serving 3D Event Display at: {url}")
    print("Press Ctrl+C to stop.")
    print("=" * 78)

    server = ThreadingHTTPServer((args.host, args.port), make_handler(catalog, html_path))
    if args.open_browser:
        threading.Timer(0.4, lambda: webbrowser.open(url)).start()

    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\nShutting down server.")
        server.server_close()


if __name__ == "__main__":
    main()
