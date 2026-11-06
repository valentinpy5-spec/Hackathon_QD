"""V2 : optique physique contre méthode des moments (TM) sur les profils NACA modifiés.

    python3 -m ser.compare_po_mom          # comparaison sur des profils tests
    python3 -m ser.compare_po_mom grille   # SER MoM des 1 024 profils V2 -> resultats/ser_mom_V2.csv
"""
import csv
import os
import sys
from multiprocessing import Pool

import numpy as np

from aero.naca4 import naca4_modified, leading_edge_radius_modified
from .po2d import wavenumber, echo_width, radar_directions, to_db
from .mom2d import solve_tm, resample

K = wavenumber(10e9)
LAM = 2 * np.pi / K
HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def mom_contour(m, p, t, I, T=0.3, per_lambda=15):
    """Contour rééchantillonné : lambda/per_lambda partout, raffiné au nez (r_BA / 4)."""
    X, Y = naca4_modified(m, p, t, I, T, n_half=600)
    h = LAM / per_lambda
    r = leading_edge_radius_modified(t, I)
    refine = lambda x: min(1.0, max((r / 4) / h, x / 0.03))   # r/4 au nez, lambda/per_lambda ailleurs
    return resample(X, Y, h, refine)


def ser_both(m, p, t, I, alpha, lo, hi, n=41, per_lambda=15):
    """SER moyenne du secteur [lo, hi] par PO et par MoM (même directions, même géométrie)."""
    eps = np.linspace(lo, hi, n)
    s = radar_directions(alpha, eps)
    Xp, Yp = naca4_modified(m, p, t, I, 0.3, n_half=200)
    po = echo_width(Xp, Yp, s, K)
    Xm, Ym = mom_contour(m, p, t, I, per_lambda=per_lambda)
    mo = solve_tm(Xm, Ym, K, s)
    return float(to_db(po.mean())), float(to_db(mo.mean())), len(Xm) - 1


def _grid_one(row):
    m, p, t, I, a = (float(row[k]) for k in ("m", "p", "t", "I", "alpha"))
    a = a if np.isfinite(a) else 0.0
    po1, mo1, n = ser_both(m, p, t, I, a, -20, 20)
    po2, mo2, _ = ser_both(m, p, t, I, a, -60, 0)
    return dict(im=row["im"], ip=row["ip"], it=row["it"], iI=row["iI"], SER_S1_PO=po1, SER_S1_MoM=mo1,
                SER_S2_PO=po2, SER_S2_MoM=mo2, N=n)


if __name__ == "__main__":
    if len(sys.argv) > 1 and sys.argv[1] == "grille":
        rows = list(csv.DictReader(open(os.path.join(HERE, "resultats", "base_V2.csv"))))
        with Pool(9) as pool:
            out = pool.map(_grid_one, rows, chunksize=4)
        path = os.path.join(HERE, "resultats", "ser_mom_V2.csv")
        with open(path, "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=list(out[0].keys()))
            w.writeheader()
            w.writerows(out)
        d1 = np.array([r["SER_S1_MoM"] - r["SER_S1_PO"] for r in out])
        print(f"{len(out)} profils ; MoM - PO (S1) : moyenne {d1.mean():+.2f} dB, écart-type {d1.std():.2f} dB -> {path}")
    else:
        print(f"{'t':>4} {'I':>3} {'kr':>5} {'PO S1':>7} {'MoM S1':>7} {'MoM fin':>7} {'N':>5}")
        for t in (0.08, 0.12, 0.16):
            for I in (2.0, 4.0, 6.0, 8.0):
                po, mo, n = ser_both(0.0, 0.4, t, I, 0.0, -20, 20)
                _, mo_f, _ = ser_both(0.0, 0.4, t, I, 0.0, -20, 20, per_lambda=25)
                print(f"{t:4.2f} {I:3.0f} {K*leading_edge_radius_modified(t, I):5.2f} {po:7.2f} {mo:7.2f} {mo_f:7.2f} {n:5d}", flush=True)
