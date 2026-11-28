"""Figures du deck de wrap-up, générées depuis les données brutes de Code/resultats/.

Usage (depuis la racine du repo) : Code/.venv/Scripts/python.exe wrapup/build.py
Sortie : wrapup/figures/*.png et *.gif, fond transparent (PNG) ou navy (GIF), pour les slides sombres.
Aucun fichier de Code/resultats/ n'est modifié.
"""
import csv
import json
import os
import sys

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RES = os.path.join(ROOT, "Code", "resultats")
OUT = os.path.join(ROOT, "wrapup", "figures")
sys.path.insert(0, os.path.join(ROOT, "Code"))
os.makedirs(OUT, exist_ok=True)

from plot_style import plt  # style du projet (Agg, axes discrets), réglé ici pour des slides sombres
from matplotlib.animation import FuncAnimation, PillowWriter

NAVY, TXT, MUTE, GRID = "#08133A", "#C3CDE8", "#93A2CC", "#2B4590"
QUBO, CTRL, CLASS, CTX = "#9887ef", "#00a888", "#e76624", "#5F6F9F"  # palette validée (dataviz, fond navy)
plt.rcParams.update({
    "font.family": "Arial", "font.size": 18, "text.color": TXT, "axes.labelcolor": TXT,
    "xtick.color": MUTE, "ytick.color": MUTE, "xtick.labelsize": 17, "ytick.labelsize": 17,
    "axes.edgecolor": GRID, "grid.color": GRID, "grid.linewidth": 0.8, "axes.grid": True,
    "axes.facecolor": "none", "figure.facecolor": "none", "legend.frameon": False,
    "lines.linewidth": 2.5, "axes.titlesize": 19, "axes.titlecolor": TXT, "axes.titlelocation": "left",
})


def load(name):
    with open(os.path.join(RES, name), encoding="utf-8") as f:
        return json.load(f)


def rows(name):
    with open(os.path.join(RES, name), encoding="utf-8") as f:
        return list(csv.DictReader(f))


def figure(wpx, hpx, **kw):
    """Taille en pixels affichés sur la slide ; 18 pt à 200 dpi = 25 px affichés."""
    return plt.subplots(figsize=(wpx / 100, hpx / 100), **kw)


def save(fig, name):
    fig.savefig(os.path.join(OUT, name), dpi=200, transparent=True)
    plt.close(fig)
    print("->", name)


def pareto(F):
    """Indices non dominés (minimisation des deux colonnes)."""
    idx = []
    for i, f in enumerate(F):
        if not np.any(np.all(F <= f, axis=1) & np.any(F < f, axis=1)):
            idx.append(i)
    return np.array(idx)


# ------------------------------------------------------------------ 1. bord rond contre bord pointu
from aero.naca4 import naca4_modified

fig, (a1, a2) = figure(1500, 430, ncols=2, gridspec_kw=dict(width_ratios=[2.2, 1]))
for I, col, lab in ((6.0, MUTE, "bord rond (I = 6)"), (2.0, QUBO, "bord pointu (I = 2)")):
    X, Y = naca4_modified(0.02, 0.40, 0.12, I, n_half=400)
    for ax in (a1, a2):
        ax.plot(X, Y, color=col, lw=3 if I == 2 else 2.5)
    a1.text(0.62, 0.075 if I == 6 else -0.095, lab, color=col, fontsize=19)
a1.set(aspect="equal", xlim=(-0.03, 1.03), ylim=(-0.12, 0.12))
a2.set(aspect="equal", xlim=(-0.01, 0.07), ylim=(-0.04, 0.05))
a2.set_title("zoom sur le nez")
for ax in (a1, a2):
    ax.axis("off")
fig.subplots_adjust(left=0.01, right=0.99, top=0.88, bottom=0.04, wspace=0.08)
save(fig, "profils_nez.png")

# ------------------------------------------------------------------ 2. V0 : boucle sur la traînée seule
B0 = json.load(open(os.path.join(ROOT, "Code", "archives", "V0_note_intention", "resultats",
                                 "experiences_quantique.json"), encoding="utf-8"))["boucle"]
fig, ax = figure(1100, 600)
mc, rc = np.array(B0["median_curve"]), np.array(B0["random_curve"])
ax.plot(np.arange(1, len(rc) + 1), rc, color=CTX, label="hasard")
ax.plot(np.arange(1, len(mc) + 1), mc, color=QUBO, label="boucle QUBO (médiane)")
ax.axhline(B0["true_min"], color=MUTE, lw=1.5, ls="--")
ax.text(len(rc) + 12, B0["true_min"] - 0.25, "vrai minimum", color=MUTE, ha="right", va="top", fontsize=17)
ax.set(xlim=(0, len(rc) + 12), xlabel="simulations", ylabel="meilleure traînée trouvée (counts)")
ax.text(len(mc) + 4, mc[-1] + 0.2, "boucle QUBO", color=QUBO, fontsize=18, va="bottom")
ax.text(len(rc) - 1, rc[-1] + 0.3, "hasard", color=MUTE, fontsize=18, ha="right", va="bottom")
ax.spines[["top", "right"]].set_visible(False)
fig.subplots_adjust(left=0.11, right=0.97, top=0.95, bottom=0.15)
save(fig, "v0_boucle.png")

# ------------------------------------------------------------------ 3. V1 : front dégénéré (S1, modèle rapide)
R1 = [r for r in rows("base_complete_S1_note.csv") if r["feasible"] == "True"]
F1 = np.array([[float(r["CD_counts"]), float(r["SER_db"])] for r in R1])
t1 = np.array([float(r["t"]) for r in R1])
fr = pareto(F1)
print(f"   V1 S1 : {len(R1)} faisables, front {len(fr)} profils, épaisseurs du front {sorted(set(t1[fr]))}")
fig, ax = figure(862, 470)  # affichée à cette taille sur la slide : 17,5 pt = 24 px
ax.scatter(F1[:, 0], F1[:, 1], s=22, color=CTX, alpha=0.8, lw=0)
o = fr[np.argsort(F1[fr, 0])]
ax.plot(F1[o, 0], F1[o, 1], color=QUBO, lw=2)
ax.scatter(F1[fr, 0], F1[fr, 1], s=90, color=QUBO, edgecolor=NAVY, lw=2, zorder=3)
XR = 128
ax.set(xlim=(50, XR), xlabel="traînée (counts), gauche = mieux", ylabel="SER (dB), bas = mieux")
# chaque bande horizontale est une épaisseur : étiquetée à droite, alignée sur sa bande
ax.text(XR + 2, F1[:, 1].max() + 0.9, "épaisseur", color=MUTE, fontsize=17.5, va="center", clip_on=False)
for t in sorted(set(t1)):
    ax.text(XR + 2, np.median(F1[t1 == t, 1]), f"{100 * t:.0f} %", color=TXT if t == t1[fr].min() else MUTE,
            fontsize=17.5, va="center", clip_on=False)
ax.annotate(f"front de Pareto :\n{len(fr)} profils, tous à {100 * t1[fr].min():.0f} %", xy=(F1[fr, 0].max() + 1, F1[fr, 1].mean()),
            xytext=(126, F1[fr, 1].mean() + 0.9), color=QUBO, fontsize=17.5, va="center", ha="right",
            bbox=dict(facecolor=NAVY, edgecolor="none", pad=3),
            arrowprops=dict(arrowstyle="->", color=QUBO, lw=1.5))
ax.spines[["top", "right"]].set_visible(False)
fig.subplots_adjust(left=0.1, right=0.83, top=0.93, bottom=0.17)
save(fig, "v1_front.png")

# ------------------------------------------------------------------ 4. V2 : la boucle construit le front (GIF, S2 graine 0)
base = {(r["im"], r["ip"], r["it"], r["iI"]): r for r in rows("base_V2.csv")}
mom = {(r["im"], r["ip"], r["it"], r["iI"]): float(r["SER_S2_MoM"]) for r in rows("ser_mom_V2.csv")}
key = lambda idx: tuple(str(i) for i in idx)
feas = [k for k, r in base.items() if r["feasible"] == "True"]
Fv = np.array([[float(base[k]["CD_counts"]), mom[k]] for k in feas])
front_v2 = pareto(Fv)
print(f"   V2 S2 : {len(feas)} faisables, front {len(front_v2)} profils")
T = load("trace_boucle_V2.json")
steps = [[c["idx"] for c in T["init"]]] + [[c["idx"] for c in it["cand"]] for it in T["iterations"]]
front_keys = {feas[i] for i in front_v2}
fig = plt.figure(figsize=(10, 6.2), facecolor=NAVY)
ax = fig.add_axes([0.11, 0.14, 0.86, 0.78], facecolor=NAVY)
ax.scatter(Fv[:, 0], Fv[:, 1], s=14, color=CTX, alpha=0.45, lw=0)
ax.scatter(Fv[front_v2, 0], Fv[front_v2, 1], s=150, facecolor="none", edgecolor=MUTE, lw=1.5)
XMAX = 125  # un profil du front a une traînée très forte : signalé hors cadre plutôt qu'écraser l'axe
ax.set(xlabel="traînée (counts)", ylabel="SER en contrebas, S2 (dB)", xlim=(30, XMAX))
out = Fv[front_v2][Fv[front_v2, 0] > XMAX]
for cd, ser in out:
    ax.annotate(f"front : 1 profil à {cd:.0f} counts", xy=(XMAX, ser), xytext=(XMAX - 2, ser),
                ha="right", va="center", fontsize=17, color=CTRL, arrowprops=dict(arrowstyle="->", color=CTRL))
ax.spines[["top", "right"]].set_visible(False)
sim = ax.scatter([], [], s=46, color=QUBO, lw=0)
hit = ax.scatter([], [], s=150, color=CTRL, edgecolor=NAVY, lw=1.5, zorder=3)
label = ax.text(0.98, 0.95, "", transform=ax.transAxes, ha="right", va="top", fontsize=20, color=TXT)
seen = []


def frame(i):
    if i < len(steps):
        seen.extend(key(x) for x in steps[i])
    pts = np.array([[float(base[k]["CD_counts"]), mom[k]] for k in seen if k in mom and base[k]["feasible"] == "True"])
    found = [k for k in seen if k in front_keys]
    if len(pts):
        sim.set_offsets(pts)
    if found:
        hit.set_offsets(np.array([[float(base[k]["CD_counts"]), mom[k]] for k in found]))
    label.set_text(f"{len(set(seen))} simulations · front : {len(set(found))}/{len(front_keys)}")
    return sim, hit, label


n_frames = len(steps) + 8  # 8 images de pause à la fin
FuncAnimation(fig, frame, frames=n_frames, blit=False).save(
    os.path.join(OUT, "v2_boucle.gif"), writer=PillowWriter(fps=3), dpi=100,
    savefig_kwargs={"facecolor": NAVY})
plt.close(fig)
print("-> v2_boucle.gif")

# ------------------------------------------------------------------ 5. V3 L1 : simulations par graine, S1 et S2
L1 = load("pipeline_L1_V3.json")
METH = [("loop_exact", "Boucle QUBO", QUBO), ("quad_mean", "Même boucle, sans QUBO", CTRL),
        ("gp_mean", "Krigeage + minimum", CTX), ("quad_ei", "Quadratique + EI", CTX),
        ("gp_ei", "Optimisation bayésienne", CLASS), ("parego", "ParEGO", CTX)]


def full(tr):
    return next((n for n, frac, _ in tr if frac >= 1.0), None)


fig, axes = figure(1664, 600, ncols=2, sharey=True)
rng = np.random.default_rng(0)
for ax, sec in zip(axes, ("S1", "S2")):
    for j, (m, lab, col) in enumerate(METH):
        y0 = len(METH) - 1 - j
        v = [full(t) for t in L1[sec][m]]
        ok = np.array([x for x in v if x is not None], float)
        ax.scatter(ok, y0 + rng.uniform(-0.18, 0.18, len(ok)), s=34, color=col, alpha=0.85, lw=0)
        med = np.median([np.inf if x is None else x for x in v])
        ax.plot([med, med], [y0 - 0.32, y0 + 0.32], color=TXT, lw=3)
        ax.text(med, y0 + 0.36, f"{med:g}".replace(".", ","), ha="center", va="bottom", fontsize=17, color=TXT)
        n_fail = sum(x is None for x in v)
        if n_fail:
            ax.text(262, y0, f"{n_fail} échec", va="center", fontsize=17, color=MUTE)
    ax.set_title(f"{sec} · front de {L1[sec]['n_front']} profils")
    ax.set(xlim=(30, 290), xlabel="simulations pour retrouver tout le front exact")
    ax.grid(axis="y", visible=False)
    ax.spines[["top", "right", "left"]].set_visible(False)
axes[0].set_yticks(range(len(METH)), [lab for _, lab, _ in METH][::-1])
axes[0].tick_params(axis="y", length=0, labelsize=18)
fig.subplots_adjust(left=0.2, right=0.99, top=0.9, bottom=0.14, wspace=0.06)
save(fig, "l1_simulations.png")

# ------------------------------------------------------------------ 6. ablation : boucle QUBO contre contrôle, graine par graine
fig, axes = figure(1050, 600, ncols=2, sharey=True)
for ax, sec in zip(axes, ("S1", "S2")):
    a = np.array([full(t) for t in L1[sec]["loop_exact"]], float)
    b = np.array([full(t) for t in L1[sec]["quad_mean"]], float)
    for x, y in zip(a, b):
        ax.plot([0, 1], [x, y], color=CTX, lw=1.2, alpha=0.8)
    ax.plot([0, 1], [np.median(a), np.median(b)], color=TXT, lw=4)
    ax.scatter([0], [np.median(a)], s=160, color=QUBO, zorder=3, edgecolor=NAVY, lw=2)
    ax.scatter([1], [np.median(b)], s=160, color=CTRL, zorder=3, edgecolor=NAVY, lw=2)
    for xx, v in ((0, np.median(a)), (1, np.median(b))):
        ax.text(xx + (-0.08 if xx == 0 else 0.08), v, f"{v:g}".replace(".", ","), ha="right" if xx == 0 else "left",
                va="center", fontsize=19, color=TXT, fontweight="bold")
    ax.set_xticks([0, 1], ["avec QUBO", "sans QUBO"])
    ax.set(xlim=(-0.45, 1.45), title=sec)
    ax.grid(axis="x", visible=False)
    ax.spines[["top", "right"]].set_visible(False)
axes[0].set_ylabel("simulations (front complet)")
fig.subplots_adjust(left=0.12, right=0.98, top=0.9, bottom=0.1, wspace=0.15)
save(fig, "ablation.png")

# ------------------------------------------------------------------ 7. QAOA : gain sur le hasard, avec et sans contraintes
A = load("anatomie_qubo_V3.json")["qaoa"]
fig, ax = figure(900, 560)
ps = [1, 2, 3]
w = 0.36
for k, (kind, lab, col) in enumerate((("complet", "QUBO de la boucle (avec contraintes)", QUBO),
                                      ("sans_contraintes", "même objectif, sans contraintes", CTRL))):
    g = [np.median([[r["p_opt"] / r["p_random"] for r in v[kind]][i] for v in A.values()]) for i in range(3)]
    xs = np.arange(3) + (k - 0.5) * w
    ax.bar(xs, g, width=w - 0.04, color=col, label=lab)
    for x, v in zip(xs, g):
        ax.text(x, v * 1.12, f"×{v:.0f}" if v >= 10 else f"×{v:.1f}".replace(".", ","), ha="center",
                va="bottom", fontsize=17, color=TXT)
ax.set_yscale("log")
ax.set(ylim=(0.5, 2000), ylabel="probabilité de l'optimum / hasard")
ax.set_xticks(range(3), [f"QAOA p = {p}" for p in ps])
ax.legend(loc="upper left", fontsize=17)
ax.grid(axis="x", visible=False)
ax.spines[["top", "right"]].set_visible(False)
fig.subplots_adjust(left=0.14, right=0.98, top=0.97, bottom=0.1)
save(fig, "qaoa_gain.png")

# ------------------------------------------------------------------ 8. L6 : hypervolume relatif, CST 30 bits
E = load("ia_qubo_echelle.json")
ref = E["hv_ref"]
fig, ax = figure(1250, 620)
for k, lab, col in (("hybride_modele", "témoin hybride (non QUBO)", CTRL), ("quadz_modele", "boucle QUBO", QUBO),
                    ("nsga2_16", "NSGA-II", CLASS), ("hasard", "hasard", CTX)):
    g = np.array(E[k]["grille"])
    m, q1, q3 = (np.array(E[k][s]) / ref for s in ("hv_median", "hv_q25", "hv_q75"))
    ax.fill_between(g, q1, q3, color=col, alpha=0.18, lw=0)
    ax.plot(g, m, color=col)
    dy = {"hybride_modele": 0.012, "quadz_modele": -0.014}.get(k, 0)
    ax.text(g[-1] + 6, m[-1] + dy, lab, color=col if col != CTX else MUTE, va="center", fontsize=18)
ax.set(xlim=(20, 440), xlabel="simulations XFOIL", ylabel="hypervolume / front de référence")
ax.set_xticks([20, 100, 200, 300])
ax.spines[["top", "right"]].set_visible(False)
fig.subplots_adjust(left=0.1, right=0.99, top=0.96, bottom=0.14)
save(fig, "l6_hypervolume.png")

# ------------------------------------------------------------------ 9. fidélité : MoM/Mie, PO/MoM, modèle rapide/XFOIL
from ser.mom2d import mie_tm, solve_tm

fig, axes = figure(1664, 540, ncols=3)
lam = 0.03
k0 = 2 * np.pi / lam
kas = np.geomspace(0.5, 31, 160)
axes[0].plot(kas, [10 * np.log10(mie_tm(ka, ka / k0) / lam) for ka in kas], color=MUTE, label="série de Mie")
kd = np.geomspace(0.5, 31, 9)
md = []
for ka in kd:
    a = ka / k0
    n = max(64, int(15 * 2 * np.pi * a / lam))
    th = np.linspace(0, 2 * np.pi, n + 1)
    md.append(10 * np.log10(float(np.ravel(solve_tm(a * np.cos(th), a * np.sin(th), k0, np.array([[1.0, 0.0]])))[0]) / lam))
axes[0].scatter(kd, md, s=70, color=QUBO, zorder=3, label="méthode des moments")
axes[0].set(xscale="log", xlabel="ka (taille / longueur d'onde)", ylabel="SER du cylindre (dB/λ)")
axes[0].legend(fontsize=16, loc="upper left")
S = rows("ser_mom_V2.csv")
po, mm = np.array([float(r["SER_S1_PO"]) for r in S]), np.array([float(r["SER_S1_MoM"]) for r in S])
axes[1].scatter(mm, po, s=10, color=QUBO, alpha=0.5, lw=0)
lo, hi = min(mm.min(), po.min()), max(mm.max(), po.max())
axes[1].plot([lo, hi], [lo, hi], color=MUTE, lw=1.5, ls="--")
axes[1].set(xlabel="SER par méthode des moments (dB)", ylabel="SER par optique physique (dB)")
xf = {(r["im"], r["ip"], r["it"]): float(r["xf_CD"]) for r in rows("base_xfoil_a2.csv") if r["xf_ok"] == "True"}
ou = {(r["im"], r["ip"], r["it"]): float(r["CD"]) for r in rows("base_aero_Re3e+06_a2.csv")}
kk = [k for k in xf if k in ou]
cx, co = 1e4 * np.array([xf[k] for k in kk]), 1e4 * np.array([ou[k] for k in kk])
axes[2].scatter(cx, co, s=10, color=QUBO, alpha=0.5, lw=0)
axes[2].plot([cx.min(), cx.max()], [cx.min(), cx.max()], color=MUTE, lw=1.5, ls="--")
axes[2].set(xlabel="traînée XFOIL (counts)", ylabel="traînée modèle rapide (counts)")
for ax in axes:
    ax.spines[["top", "right"]].set_visible(False)
fig.subplots_adjust(left=0.06, right=0.99, top=0.96, bottom=0.17, wspace=0.32)
save(fig, "validation.png")
