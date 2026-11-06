"""La sortie de la phase 1 est-elle « QUBO-compatible » ?

La boucle quantique (phase 4) remplace le simulateur par un modèle
quadratique ajusté par moindres carrés. On mesure ici, sur la base exhaustive,
la qualité de deux familles de modèles pour CD (et CL) :

  A. quadratique en (m, p, t) normalisés : 10 coefficients, puis développé en
     bits (reste un QUBO, voir design_space.py) ;
  B. QUBO direct sur les 9 bits : 1 + 9 + 36 = 46 coefficients.

et l'effet d'un petit échantillon (ce que verra réellement la boucle itérative).

    python3 analyse_quadratique.py
"""
import csv
import os
from itertools import combinations

import numpy as np

from plot_style import plt, C_UP, C_LO, C_REF, SEQ_CMAP

HERE = os.path.dirname(__file__)
rows = list(csv.DictReader(open(os.path.join(HERE, "resultats", "base_aero_Re3e+06_a2.csv"))))
rows = [r for r in rows if r["valid"] == "True"]
idx = np.array([[int(r["im"]), int(r["ip"]), int(r["it"])] for r in rows])
bits = np.array([[int(b) for b in r["bits"]] for r in rows])
CD = np.array([float(r["CD"]) for r in rows])
CL = np.array([float(r["CL"]) for r in rows])


def feats_cont(I):
    z = I / 7.0 * 2 - 1                          # niveaux -> [-1, 1]
    cols = [np.ones(len(z))] + [z[:, i] for i in range(3)]
    cols += [z[:, i] * z[:, j] for i in range(3) for j in range(i, 3)]
    return np.column_stack(cols)


def feats_bits(B):
    cols = [np.ones(len(B))] + [B[:, i] for i in range(B.shape[1])]
    cols += [B[:, i] * B[:, j] for i, j in combinations(range(B.shape[1]), 2)]
    return np.column_stack(cols)


def fit_r2(F, y, train):
    c, *_ = np.linalg.lstsq(F[train], y[train], rcond=None)
    pred = F @ c
    r2 = 1 - np.sum((y - pred) ** 2) / np.sum((y - y.mean()) ** 2)
    return r2, pred


def spearman(a, b):
    ra, rb = np.argsort(np.argsort(a)), np.argsort(np.argsort(b))
    return np.corrcoef(ra, rb)[0, 1]


all_ = np.ones(len(CD), dtype=bool)
print(f"{len(CD)} profils valides")
for name, y in (("CD", CD), ("CL", CL)):
    for fam, F in (("A quadratique (m,p,t), 10 coef", feats_cont(idx)),
                   ("B QUBO sur 9 bits, 46 coef", feats_bits(bits))):
        r2, _ = fit_r2(F, y, all_)
        print(f"{name}  {fam:32s}  R2 (base complète) = {r2:.4f}")

print("\nPetits échantillons (ce que voit la boucle QUBO), 200 tirages :")
rng = np.random.default_rng(0)
F = feats_cont(idx)
for n in (12, 20, 30, 50):
    r2s, rhos, top = [], [], []
    for _ in range(200):
        tr = np.zeros(len(CD), dtype=bool)
        tr[rng.choice(len(CD), n, replace=False)] = True
        r2, pred = fit_r2(F, CD, tr)
        r2s.append(r2)
        rhos.append(spearman(pred, CD))
        # le meilleur profil prédit est-il dans le top 5 % réel ?
        top.append(np.argsort(np.argsort(CD))[np.argmin(pred)] < 0.05 * len(CD))
    print(f"n={n:3d}  R2 médian {np.median(r2s):.3f}   rho de Spearman médian {np.median(rhos):.3f}"
          f"   meilleur prédit dans le top 5 % : {100*np.mean(top):.0f} %")

# ---------------------------------------------------------------- figures
r2, pred = fit_r2(F, CD, all_)
fig, ax = plt.subplots(1, 3, figsize=(12, 3.6))
ax[0].plot(CD * 1e4, pred * 1e4, "o", ms=3, color=C_UP, alpha=0.6)
lim = [CD.min() * 1e4, CD.max() * 1e4]
ax[0].plot(lim, lim, "--", color=C_REF, lw=1)
ax[0].set(xlabel="CD modèle aéro (counts)", ylabel="CD quadratique (counts)",
          title=f"Modèle A sur toute la base, R² = {r2:.3f}")

# carte CD(m, t) à p = 0.40 (ip = 4)
grid = np.full((8, 8), np.nan)
gridL = np.full((8, 8), np.nan)
for (im, ip, it), cd, cl in zip(idx, CD, CL):
    if ip == 4:
        grid[it, im] = cd * 1e4
        gridL[it, im] = cl
ext = [-0.5, 7.5, 5, 21]
im0 = ax[1].imshow(grid, origin="lower", cmap=SEQ_CMAP, aspect="auto", extent=ext)
ax[1].set(xlabel="cambrure m (%)", ylabel="épaisseur t (%)", title="CD (counts), p = 0.40, alpha = 2°")
ax[1].grid(False)
fig.colorbar(im0, ax=ax[1])
im1 = ax[2].imshow(gridL, origin="lower", cmap=SEQ_CMAP, aspect="auto", extent=ext)
ax[2].set(xlabel="cambrure m (%)", ylabel="épaisseur t (%)", title="CL, p = 0.40, alpha = 2°")
ax[2].grid(False)
fig.colorbar(im1, ax=ax[2])
fig.tight_layout()
fig.savefig(os.path.join(HERE, "figures", "analyse_quadratique.png"))
