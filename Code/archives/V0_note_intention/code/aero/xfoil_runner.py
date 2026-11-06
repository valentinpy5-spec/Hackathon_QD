"""Pilotage d'XFOIL en boîte noire pour valider le modèle de la phase 1.

XFOIL (Drela, MIT) résout la même physique avec un couplage visqueux FORT
(panneaux à tourbillon linéaire + couche limite intégrale à deux équations +
transition e^N + sillage). C'est la référence « basse fidélité de qualité »
contre laquelle notre couplage faible doit être comparé.

Nécessite l'exécutable `xfoil` dans le PATH (paquet `xfoil` sous Debian/Ubuntu).
"""
import os
import shutil
import subprocess
import tempfile

import numpy as np

from .naca4 import naca4


def xfoil_available(exe="xfoil"):
    return shutil.which(exe) is not None


def run_xfoil(m, p, t, alphas, Re, ncrit=9.0, x_trip=(1.0, 1.0), n_iter=200,
              exe="xfoil", timeout=60):
    """Polaire XFOIL. Renvoie un dict de tableaux (alpha, CL, CD, CDp, CM, xtr_up, xtr_lo).

    Les incidences non convergées sont absentes du résultat (XFOIL ne les écrit
    pas dans la polaire) : comparer les tableaux d'alpha avant d'interpréter.
    """
    X, Y = naca4(m, p, t, n_half=120)
    with tempfile.TemporaryDirectory() as d:
        coord = os.path.join(d, "foil.dat")
        polar = os.path.join(d, "polar.txt")
        with open(coord, "w") as f:
            f.write(f"NACA_m{m:.3f}_p{p:.3f}_t{t:.3f}\n")
            for x, y in zip(X, Y):
                f.write(f"{x:.7f} {y:.7f}\n")
        cmds = ["PLOP", "G F", "", f"LOAD {coord}", "PANE", "OPER",
                f"VISC {Re:.0f}", f"ITER {n_iter}", "VPAR", f"N {ncrit}",
                f"XTR {x_trip[0]} {x_trip[1]}", "", "PACC", polar, ""]
        cmds += [f"ALFA {a:.3f}" for a in alphas]
        cmds += ["PACC", "", "QUIT"]
        subprocess.run([exe], input="\n".join(cmds) + "\n", text=True,
                       capture_output=True, timeout=timeout)
        return parse_polar(polar)


def parse_polar(path):
    keys = ["alpha", "CL", "CD", "CDp", "CM", "xtr_up", "xtr_lo"]
    rows = []
    if os.path.exists(path):
        with open(path) as f:
            started = False
            for line in f:
                if line.strip().startswith("------"):
                    started = True
                    continue
                if started and line.strip():
                    rows.append([float(v) for v in line.split()[:7]])
    arr = np.array(rows) if rows else np.empty((0, 7))
    return {k: arr[:, i] for i, k in enumerate(keys)}
