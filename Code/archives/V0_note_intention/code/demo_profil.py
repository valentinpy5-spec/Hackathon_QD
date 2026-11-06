"""Démonstration sur un profil : géométrie, Cp, couche limite, polaire, export STL.

    python3 demo_profil.py            # NACA 2412, Re 3e6
    python3 demo_profil.py 4415 6e6   # autre profil / Reynolds
"""
import os
import sys

import numpy as np

from plot_style import plt, C_UP, C_LO, INK
from aero import naca4, evaluate
from aero.geometry_export import extrude, write_stl, write_metadata

HERE = os.path.dirname(__file__)
FIG = os.path.join(HERE, "figures")
OUT = os.path.join(HERE, "resultats")
os.makedirs(FIG, exist_ok=True)
os.makedirs(OUT, exist_ok=True)

code = sys.argv[1] if len(sys.argv) > 1 else "2412"
Re = float(sys.argv[2]) if len(sys.argv) > 2 else 3e6
m, p, t = int(code[0]) / 100, int(code[1]) / 10, int(code[2:]) / 100
alpha = 2.0

r, inv, (bu, bl) = evaluate(m, p, t, alpha=alpha, Re=Re, return_details=True)
print(f"NACA {code}  alpha={alpha}  Re={Re:.1e}")
print(f"CL={r.CL:.4f}  CD={r.CD:.5f} (frottement {r.CD_f:.5f}, forme {r.CD - r.CD_f:.5f})"
      f"  CM={r.CM:.4f}  L/D={r.CL / r.CD:.1f}")
print(f"transition extrados x={r.x_tr_up:.3f} ({r.tr_mode_up}), intrados x={r.x_tr_lo:.3f} ({r.tr_mode_lo})")
print(f"valide={r.valid}  drapeaux={r.flags}")

# ---------------------------------------------------------------- figure 1
fig, ax = plt.subplots(2, 2, figsize=(10, 6.5))
P = inv.panels
ax[0, 0].plot(np.r_[P.xs, P.xe[-1]], np.r_[P.ys, P.ye[-1]], color=INK, lw=1.2)
ax[0, 0].set(aspect="equal", xlabel="x/c", ylabel="y/c", title=f"NACA {code} ({P.n} panneaux)")

ax[0, 1].plot(bu.x, 1 - bu.Ue**2, color=C_UP, label="extrados")
ax[0, 1].plot(bl.x, 1 - bl.Ue**2, color=C_LO, label="intrados")
ax[0, 1].invert_yaxis()
ax[0, 1].set(xlabel="x/c", ylabel="Cp", title=f"Cp non visqueux, alpha = {alpha}°")
ax[0, 1].legend()

ax[1, 0].plot(bu.x, bu.theta * 1e3, color=C_UP, label="extrados")
ax[1, 0].plot(bl.x, bl.theta * 1e3, color=C_LO, label="intrados")
for b, c in ((bu, C_UP), (bl, C_LO)):
    if np.isfinite(b.x_tr):
        ax[1, 0].axvline(b.x_tr, color=c, lw=1, ls=":")
ax[1, 0].set(xlabel="x/c", ylabel=r"$\theta/c \times 10^3$", title="Épaisseur de quantité de mouvement")
ax[1, 0].legend()

ax[1, 1].plot(bu.x, bu.H, color=C_UP, label="H extrados")
ax[1, 1].plot(bl.x, bl.H, color=C_LO, label="H intrados")
ax[1, 1].set(xlabel="x/c", ylabel="H", title="Facteur de forme (saut à la transition)")
ax[1, 1].legend()
fig.tight_layout()
fig.savefig(os.path.join(FIG, f"demo_NACA{code}.png"))

# ---------------------------------------------------------------- polaire
alphas = np.arange(-4, 11, 1.0)
res = [evaluate(m, p, t, alpha=a, Re=Re) for a in alphas]
CL = np.array([x.CL for x in res])
CD = np.array([x.CD for x in res])
ok = np.array([x.valid for x in res])
fig, ax = plt.subplots(1, 2, figsize=(9, 3.6))
ax[0].plot(alphas, CL, color=C_UP)
ax[0].set(xlabel="alpha (°)", ylabel="CL", title="Portance (non visqueuse)")
ax[1].plot(CD[ok] * 1e4, CL[ok], "o-", color=C_UP, ms=4, label="attaché")
if (~ok).any():
    ax[1].plot(CD[~ok] * 1e4, CL[~ok], "x", color=C_LO, ms=7, label="décollement turbulent")
ax[1].set(xlabel="CD (counts)", ylabel="CL", title=f"Polaire, Re = {Re:.0e}")
ax[1].legend()
fig.tight_layout()
fig.savefig(os.path.join(FIG, f"polaire_NACA{code}.png"))

# ---------------------------------------------------------------- export SER
X, Y = naca4(m, p, t, n_half=100)
T = extrude(X, Y, span=1.0)
stl = os.path.join(OUT, f"NACA{code}.stl")
write_stl(T, stl, name=f"NACA{code}")
write_metadata(stl.replace(".stl", ".json"), r, span=1.0, n_tri=len(T))
print(f"STL : {stl}  ({len(T)} triangles)")
