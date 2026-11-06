"""Validation V5 étendue : XFOIL sur les 512 profils de la grille, comparé à notre modèle.

Questions posées (ce qui compte pour l'optimisation) :
    1. les deux modèles classent-ils les profils de la même façon ? (corrélation de rang)
    2. donnent-ils le même front de Pareto traînée / SER ?

    python3 validation_xfoil_grille.py          # environ 10 min sur 8 cœurs
Sortie : resultats/base_xfoil_a2.csv, puis un résumé imprimé.
"""
import csv
import os
import sys
from multiprocessing import Pool

import numpy as np

from aero.design_space import DesignSpace
from aero.xfoil_runner import run_xfoil, xfoil_available

HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(HERE, "resultats", "base_xfoil_a2.csv")
ALPHA, RE = 2.0, 3e6


def one(idx):
    sp = DesignSpace()
    m, p, t = sp.values(idx)
    r = run_xfoil(m, p, t, [ALPHA], RE, timeout=90)
    ok = len(r["alpha"]) == 1
    return dict(im=idx[0], ip=idx[1], it=idx[2], m=m, p=p, t=t,
                xf_ok=ok, xf_CL=float(r["CL"][0]) if ok else np.nan, xf_CD=float(r["CD"][0]) if ok else np.nan,
                xf_CM=float(r["CM"][0]) if ok else np.nan,
                xf_xtr_up=float(r["xtr_up"][0]) if ok else np.nan, xf_xtr_lo=float(r["xtr_lo"][0]) if ok else np.nan)


def spearman(a, b):
    ra, rb = np.argsort(np.argsort(a)), np.argsort(np.argsort(b))
    return float(np.corrcoef(ra, rb)[0, 1])


if __name__ == "__main__":
    if not xfoil_available():
        sys.exit("XFOIL introuvable")
    if not os.path.exists(OUT):
        idxs = list(DesignSpace().all_indices())
        with Pool(8) as pool:
            rows = pool.map(one, idxs, chunksize=4)
        with open(OUT, "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
            w.writeheader()
            w.writerows(rows)
    xf = {(int(r["im"]), int(r["ip"]), int(r["it"])): r for r in csv.DictReader(open(OUT))}
    ours = {(int(r["im"]), int(r["ip"]), int(r["it"])): r
            for r in csv.DictReader(open(os.path.join(HERE, "resultats", "base_aero_Re3e+06_a2.csv")))}
    keys = [k for k in ours if xf[k]["xf_ok"] == "True" and ours[k]["valid"] == "True"]
    cd_o = np.array([float(ours[k]["CD"]) for k in keys])
    cd_x = np.array([float(xf[k]["xf_CD"]) for k in keys])
    cl_o = np.array([float(ours[k]["CL"]) for k in keys])
    cl_x = np.array([float(xf[k]["xf_CL"]) for k in keys])
    print(f"XFOIL convergé sur {sum(r['xf_ok'] == 'True' for r in xf.values())} / 512 profils")
    print(f"C_D : écart moyen {100 * np.mean(cd_o / cd_x - 1):+.1f} %, rang de Spearman {spearman(cd_o, cd_x):.3f}")
    print(f"C_L : écart moyen {100 * np.mean(cl_o / cl_x - 1):+.1f} %, rang de Spearman {spearman(cl_o, cl_x):.3f}")
