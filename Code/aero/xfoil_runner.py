"""Pilotage d'XFOIL en boîte noire pour valider le modèle de la phase 1.

XFOIL (Drela, MIT) résout la même physique avec un couplage visqueux FORT
(panneaux à tourbillon linéaire + couche limite intégrale à deux équations +
transition e^N + sillage). C'est la référence « basse fidélité de qualité »
contre laquelle notre couplage faible doit être comparé.

Exécutable : `xfoil` dans le PATH, sinon la copie locale extraite du paquet Debian
dans Code/.tools/xfoil_pkg (apt-get download xfoil ; dpkg -x ...), sans droits
administrateur.

Affichage : cette compilation d'XFOIL n'aboutit en visqueux que si sa sortie
graphique est active (sinon exception flottante) ; il lui faut donc un écran X.
Sans écran, on démarre un écran virtuel Xvfb (copie locale dans Code/.tools/xvfb_pkg).
"""
import os
import shutil
import subprocess
import tempfile

import numpy as np

from .naca4 import naca4


TOOLS = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), ".tools")
LOCAL_XFOIL = os.path.join(TOOLS, "xfoil_pkg", "usr", "bin", "xfoil")
LOCAL_XVFB = os.path.join(TOOLS, "xvfb_pkg", "usr", "bin", "Xvfb")
XVFB_DISPLAY = ":97"


def find_xfoil(exe="xfoil"):
    path = shutil.which(exe)
    if path:
        return path
    return LOCAL_XFOIL if os.path.exists(LOCAL_XFOIL) else None


def xfoil_available(exe="xfoil"):
    return find_xfoil(exe) is not None


def xfoil_env():
    """Environnement d'exécution avec un écran X (réel si présent, sinon Xvfb local)."""
    env = dict(os.environ)
    if env.get("DISPLAY"):
        return env
    xvfb = shutil.which("Xvfb") or (LOCAL_XVFB if os.path.exists(LOCAL_XVFB) else None)
    if xvfb is None:
        raise RuntimeError("ni écran X ni Xvfb : XFOIL ne peut pas tourner en visqueux")
    sock = f"/tmp/.X11-unix/X{XVFB_DISPLAY[1:]}"
    if not os.path.exists(sock):
        subprocess.Popen([xvfb, XVFB_DISPLAY, "-screen", "0", "1024x768x24", "-nolisten", "tcp"],
                         stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, start_new_session=True)
        for _ in range(50):
            if os.path.exists(sock):
                break
            import time
            time.sleep(0.1)
    env["DISPLAY"] = XVFB_DISPLAY
    return env


def run_xfoil(m, p, t, alphas, Re, ncrit=9.0, x_trip=(1.0, 1.0), n_iter=200,
              exe="xfoil", timeout=60, coords=None, cls=None):
    """Polaire XFOIL. Renvoie un dict de tableaux (alpha, CL, CD, CDp, CM, xtr_up, xtr_lo).

    Les incidences non convergées sont absentes du résultat (XFOIL ne les écrit
    pas dans la polaire) : comparer les tableaux d'alpha avant d'interpréter.
    """
    # V2 : coords=(X, Y) permet de passer une autre géométrie (série modifiée) ;
    # cls=[...] demande des points à portance imposée (commande CL) au lieu d'incidences.
    X, Y = coords if coords is not None else naca4(m, p, t, n_half=120)
    with tempfile.TemporaryDirectory() as d:
        coord = os.path.join(d, "foil.dat")
        polar = os.path.join(d, "polar.txt")
        with open(coord, "w") as f:
            f.write(f"NACA_m{m:.3f}_p{p:.3f}_t{t:.3f}\n")
            for x, y in zip(X, Y):
                f.write(f"{x:.7f} {y:.7f}\n")
        cmds = [f"LOAD {coord}", "PANE", "OPER",
                f"VISC {Re:.0f}", f"ITER {n_iter}", "VPAR", f"N {ncrit}",
                f"XTR {x_trip[0]} {x_trip[1]}", "", "PACC", polar, ""]
        cmds += [f"CL {c:.4f}" for c in (cls or [])]
        cmds += [f"ALFA {a:.3f}" for a in alphas]
        cmds += ["PACC", "", "QUIT"]
        try:
            subprocess.run([find_xfoil(exe)], input="\n".join(cmds) + "\n", text=True,
                           capture_output=True, timeout=timeout, env=xfoil_env())
        except subprocess.TimeoutExpired:
            pass   # XFOIL bloqué (point non convergé) : on garde les points déjà écrits
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
