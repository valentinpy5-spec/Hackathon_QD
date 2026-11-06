"""Figures de l'itération V2 du cours aéro : série NACA modifiée et XFOIL de référence.

Usage (depuis le dossier du cours) : python3 tools/make_figures_v2.py
Lit Code/resultats/base_V2.csv (produit par python3 -m optim.problem_v2).
"""
import json
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
PROJ = os.path.abspath(os.path.join(HERE, "..", ".."))
CODE = os.path.join(PROJ, "Code")
sys.path.insert(0, CODE)

from aero.naca4 import naca4_modified, thickness_modified, leading_edge_radius_modified  # noqa: E402
from optim.problem_v2 import build_table_v2  # noqa: E402

FIG = os.path.join(HERE, "figures")
NUM = {}
STY = {2.0: "corange", 4.0: "cgreen", 6.0: "cblue", 8.0: "cmuted"}


def write(name, body):
    with open(os.path.join(FIG, name + ".tex"), "w") as f:
        f.write(f"% figure générée par tools/make_figures_v2.py : {name}\n{body}\n")


def co(xs, ys, fmt="{:.5g}"):
    return " ".join(f"({fmt.format(float(x))},{fmt.format(float(y))})" for x, y in zip(xs, ys)
                    if np.isfinite(x) and np.isfinite(y))


def fig_formes():
    pl, zl = [], []
    for I, st in STY.items():
        X, Y = naca4_modified(0.02, 0.4, 0.12, I, 0.3, n_half=150)
        pl.append(f"\\addplot[{st},very thick] coordinates {{{co(X, Y)}}};\\addlegendentry{{$I={I:g}$, $r_{{BA}}={100*leading_edge_radius_modified(0.12, I):.2f}$~\\%}}")
        k = X < 0.06
        zl.append(f"\\addplot[{st},very thick] coordinates {{{co(X[k], Y[k])}}};")
    write("v2_formes", rf"""\begin{{tikzpicture}}
\begin{{groupplot}}[group style={{group size=2 by 1,horizontal sep=1.4cm}},title style={{font=\small}},tick label style={{font=\scriptsize}}]
\nextgroupplot[width=0.62\linewidth,height=4.2cm,axis equal image,xmin=-0.02,xmax=1.02,ymin=-0.08,ymax=0.12,
  xlabel={{$x/c$}},ylabel={{$y/c$}},title={{NACA 2412 modifié, quatre rayons de bord d'attaque}},
  legend style={{draw=none,font=\scriptsize,at={{(0.5,-0.45)}},anchor=north}},legend columns=2]
{chr(10).join(pl)}
\nextgroupplot[width=0.36\linewidth,height=5cm,axis equal image,xmin=-0.002,xmax=0.05,ymin=-0.035,ymax=0.04,
  xlabel={{$x/c$}},title={{zoom sur le nez}},xticklabel style={{/pgf/number format/fixed}}]
{chr(10).join(zl)}
\end{{groupplot}}
\end{{tikzpicture}}""")


def by_I(T, key, t_vals=(0.08, 0.10, 0.12, 0.14, 0.16, 0.18, 0.20), only_valid=True):
    out = {}
    for I in STY:
        med = []
        for t in t_vals:
            v = [r[key] for r in T if r["I"] == I and abs(r["t"] - t) < 1e-9 and (r["valid"] or not only_valid)
                 and np.isfinite(r[key])]
            med.append(np.median(v) if v else np.nan)
        out[I] = med
    return np.array(t_vals), out


def fig_xfoil():
    T = build_table_v2()
    t, cd = by_I(T, "CD_counts")
    _, clm = by_I(T, "CLmax", only_valid=False)
    pl1 = [f"\\addplot[{STY[I]},very thick,mark=*,mark size=1.2pt] coordinates {{{co(100 * t, cd[I])}}};\\addlegendentry{{$I={I:g}$}}" for I in STY]
    pl2 = [f"\\addplot[{STY[I]},very thick,mark=*,mark size=1.2pt] coordinates {{{co(100 * t, clm[I])}}};" for I in STY]
    write("v2_xfoil", rf"""\begin{{tikzpicture}}
\begin{{groupplot}}[group style={{group size=2 by 1,horizontal sep=1.6cm}},width=0.48\linewidth,height=5.4cm,
  xlabel={{épaisseur $t$ (\%)}},title style={{font=\small}},tick label style={{font=\scriptsize}},legend style={{draw=none,font=\scriptsize}}]
\nextgroupplot[title={{$C_D$ de croisière ($C_L=0{{,}}4$), médiane sur $m,p$}},ylabel={{counts}},legend pos=north west]
{chr(10).join(pl1)}
\nextgroupplot[title={{$C_{{L,\max}}$ (rampe 4--14°), médiane sur $m,p$}},ymin=0.6]
{chr(10).join(pl2)}
\addplot[cink,dashed] coordinates {{(8,1.2) (20,1.2)}};
\node[font=\scriptsize,anchor=south east] at (axis cs:20,1.2) {{exigence de manœuvre}};
\end{{groupplot}}
\end{{tikzpicture}}""")
    NUM["n_valid"] = sum(r["valid"] for r in T)
    NUM["n_feas"] = sum(r["feasible"] for r in T)
    NUM["clmax_fail_by_I"] = {f"{I:g}": sum(1 for r in T if r["I"] == I and r["valid"] and r["t"] >= 0.08 and not r["feasible"]) for I in STY}
    NUM["cruise_fail_by_I"] = {f"{I:g}": sum(1 for r in T if r["I"] == I and not r["valid"]) for I in STY}
    NUM["cd_med_t12"] = {f"{I:g}": float(cd[I][2]) for I in STY}
    NUM["clmax_med_t12"] = {f"{I:g}": float(clm[I][2]) for I in STY}


if __name__ == "__main__":
    fig_formes()
    fig_xfoil()
    path = os.path.join(FIG, "chiffres_v2.json")
    with open(path, "w") as f:
        json.dump(NUM, f, indent=1, default=float)
    print(json.dumps(NUM, indent=1, default=float))
