"""Vérification et validation du modèle aéro de la phase 1.

V1  panneaux : convergence en maillage, CL(Cp) = CL(Kutta-Joukowski), CD_p -> 0
V2  Thwaites sur plaque plane contre Blasius
V3  Head sur plaque plane contre la corrélation turbulente en 1/7
V4  sensibilité de CD aux paramètres numériques (n_half, x_stop)
V5  comparaison à XFOIL (si l'exécutable est installé)

Lancer depuis ce dossier :  python3 validation.py
"""
import os

import numpy as np

from plot_style import plt, C_UP, C_LO, C_REF
from aero import naca4_from_code, solve, evaluate
from aero.boundary_layer import march_surface
from aero.xfoil_runner import xfoil_available, run_xfoil

FIG = os.path.join(os.path.dirname(__file__), "figures")
os.makedirs(FIG, exist_ok=True)


def v1_panels():
    print("\n=== V1 : méthode des panneaux ===")
    print(f"{'profil':>6} {'alpha':>5} {'n_pan':>6} {'CL(Cp)':>8} {'CL(KJ)':>8} {'CD_p':>9} {'CM_c/4':>8}")
    for code, a in [("0012", 5.0), ("2412", 0.0), ("4412", 4.0)]:
        for n in (25, 50, 100, 200):
            X, Y = naca4_from_code(code, n_half=n)
            s = solve(X, Y, a)
            print(f"{code:>6} {a:5.1f} {2*n:6d} {s.CL:8.4f} {s.CL_kj:8.4f} {s.CD_p:9.2e} {s.CM:8.4f}")
    # Pente de portance : théorie des profils minces = 2 pi / rad
    X, Y = naca4_from_code("0012", n_half=100)
    cache = {}
    a = np.linspace(-4, 8, 7)
    cl = [solve(X, Y, ai, cache=cache).CL for ai in a]
    slope = np.polyfit(np.deg2rad(a), cl, 1)[0]
    print(f"pente dCL/dalpha NACA 0012 = {slope:.3f} /rad  (profil mince : {2*np.pi:.3f},"
          f" l'épaisseur l'augmente d'environ 0.77 t)")


def v2_v3_flat_plate():
    print("\n=== V2/V3 : plaque plane (Ue = 1) ===")
    Re = 1e7
    s = np.linspace(1e-4, 1.0, 400)
    Ue = np.ones_like(s)
    lam = march_surface(s, s, Ue, Re, free_transition=False, x_stop=2.0)
    Rex = Re * s
    th_blasius = 0.664 * s / np.sqrt(Rex)
    err = np.max(np.abs(lam.theta[50:] / th_blasius[50:] - 1))
    print(f"Thwaites / Blasius : écart max sur theta = {100*err:.2f} %  (attendu ~1 %, 0.671 contre 0.664)")
    print(f"H laminaire = {lam.H[-1]:.3f}  (Blasius 2.59)")

    tur = march_surface(s, s, Ue, Re, x_trip=0.0, free_transition=False, x_stop=2.0)
    th_17 = 0.036 * s * Rex ** -0.2
    i = s > 0.3
    err_t = np.max(np.abs(tur.theta[i] / th_17[i] - 1))
    print(f"Head / loi 1/7 : écart max sur theta (x > 0.3) = {100*err_t:.1f} %  (H final {tur.H[-1]:.3f})")

    fig, ax = plt.subplots(1, 2, figsize=(9, 3.4))
    ax[0].plot(s, lam.theta * 1e3, color=C_UP, label="Thwaites")
    ax[0].plot(s, th_blasius * 1e3, "--", color=C_REF, label="Blasius")
    ax[0].set(xlabel="x/c", ylabel=r"$\theta/c \times 10^3$", title="Laminaire, Re = 1e7")
    ax[0].legend()
    ax[1].plot(s, tur.theta * 1e3, color=C_LO, label="Head")
    ax[1].plot(s, th_17 * 1e3, "--", color=C_REF, label="loi 1/7")
    ax[1].set(xlabel="x/c", title="Turbulent dès le bord d'attaque")
    ax[1].legend()
    fig.tight_layout()
    fig.savefig(os.path.join(FIG, "V2_V3_plaque_plane.png"))


def v4_sensitivity():
    print("\n=== V4 : sensibilité aux paramètres numériques (NACA 2412, alpha 2, Re 3e6) ===")
    for n in (60, 100, 150, 200):
        r = evaluate(0.02, 0.4, 0.12, alpha=2, Re=3e6, n_half=n)
        print(f"n_half={n:4d}  CL={r.CL:.4f}  CD={r.CD:.5f}  xtr_up={r.x_tr_up:.3f}")
    for xs in (0.97, 0.98, 0.985, 0.99):
        r = evaluate(0.02, 0.4, 0.12, alpha=2, Re=3e6, x_stop=xs)
        print(f"x_stop={xs:5.3f}  CD={r.CD:.5f}")


def v5_xfoil():
    print("\n=== V5 : comparaison XFOIL ===")
    if not xfoil_available():
        print("xfoil absent du PATH : installer XFOIL (paquet `xfoil`) puis relancer.")
        return
    Re = 3e6
    alphas = np.arange(-2, 9, 1.0)
    for (m, p, t) in [(0.0, 0.0, 0.12), (0.02, 0.4, 0.12), (0.04, 0.4, 0.15)]:
        xf = run_xfoil(m, p, t, alphas, Re)
        ours = [evaluate(m, p, t, alpha=a, Re=Re) for a in xf["alpha"]]
        print(f"\nNACA m={m} p={p} t={t}")
        print(f"{'alpha':>5} {'CL xf':>7} {'CL ici':>7} {'CD xf':>8} {'CD ici':>8} {'xtr xf':>7} {'xtr ici':>7}")
        for a, cl, cd, xt, r in zip(xf["alpha"], xf["CL"], xf["CD"], xf["xtr_up"], ours):
            print(f"{a:5.1f} {cl:7.3f} {r.CL:7.3f} {cd:8.5f} {r.CD:8.5f} {xt:7.3f} {r.x_tr_up:7.3f}")
        # même transition qu'XFOIL : isole l'erreur couche limite + Squire-Young
        same = [evaluate(m, p, t, alpha=a, Re=Re, x_trip=(xu, xl), free_transition=False).CD
                for a, xu, xl in zip(xf["alpha"], xf["xtr_up"], xf["xtr_lo"])]
        e = np.array(same) / xf["CD"] - 1
        print(f"à transition imposée = XFOIL : écart CD moyen {100*np.mean(e):+.1f} %, max {100*np.max(np.abs(e)):.1f} %")


if __name__ == "__main__":
    v1_panels()
    v2_v3_flat_plate()
    v4_sensitivity()
    v5_xfoil()
