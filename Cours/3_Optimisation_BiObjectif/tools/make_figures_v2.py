"""Figures V2 du cours d'optimisation : problème reformulé (table XFOIL, 1 024 profils).

Usage (depuis le dossier du cours) : python3 tools/make_figures_v2.py
Lit Code/resultats/base_V2.csv et, s'il existe, Code/resultats/pipeline_V2.json.
"""
import json
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
PROJ = os.path.abspath(os.path.join(HERE, "..", ".."))
RES = os.path.join(PROJ, "Code", "resultats")
sys.path.insert(0, os.path.join(PROJ, "Code"))

from optim.problem_v2 import build_table_v2, with_objective  # noqa: E402
from optim.pareto import pareto_front  # noqa: E402

FIG = os.path.join(HERE, "figures")
STY = {2.0: "corange", 4.0: "cgreen", 6.0: "cblue", 8.0: "cmuted"}
TITLES = {"S1": "S1 : approche de face", "S2": "S2 : radar en contrebas"}
NUM = {}


def write(name, body):
    with open(os.path.join(FIG, name + ".tex"), "w") as f:
        f.write(f"% figure générée par tools/make_figures_v2.py : {name}\n{body}\n")


def co(xs, ys):
    return " ".join(f"({float(x):.5g},{float(y):.5g})" for x, y in zip(xs, ys) if np.isfinite(x) and np.isfinite(y))


def fig_fronts():
    T0 = build_table_v2()
    blocks = []
    for sec in ("S1", "S2"):
        T = with_objective(T0, sec, "MoM")
        feas = [r for r in T if r["feasible"]]
        F = np.array([[r["CD_counts"], r["SER_db"]] for r in feas])
        fr = sorted(pareto_front(F), key=lambda i: F[i, 0])
        pl = []
        for I, st in STY.items():
            k = [i for i, r in enumerate(feas) if r["I"] == I]
            pl.append(f"\\addplot[only marks,mark=*,mark size=0.9pt,{st}!70] coordinates {{{co(F[k, 0], F[k, 1])}}};"
                      + (f"\\addlegendentry{{$I={I:g}$}}" if sec == "S1" else ""))
        pl.append(f"\\addplot[cink,very thick,mark=*,mark size=1.8pt,mark options={{fill=white}},forget plot] coordinates {{{co(F[fr, 0], F[fr, 1])}}};")
        for i in fr:
            r = feas[i]
            if F[i, 0] < 120:
                pl.append(f"\\node[font=\\tiny,anchor=south west,inner sep=1pt] at (axis cs:{F[i,0]:.2f},{F[i,1]:.3f}) {{$t{100*r['t']:.0f}\\,I{r['I']:g}$}};")
        blocks.append(rf"""\nextgroupplot[title={{{TITLES[sec]}}}]
{chr(10).join(pl)}""")
        NUM[sec] = dict(n_feas=len(feas), n_front=len(fr),
                        front=[{k: feas[i][k] for k in ("m", "p", "t", "I", "CD_counts", "SER_db", "CLmax", "alpha")} for i in fr])
    write("v2_fronts", rf"""\begin{{tikzpicture}}
\begin{{groupplot}}[group style={{group size=2 by 1,horizontal sep=1.5cm}},width=0.5\linewidth,height=6.2cm,
  xlabel={{$C_D$ de croisière (counts, XFOIL)}},ylabel={{SER (dBm, MoM)}},xmin=30,xmax=120,
  title style={{font=\small}},tick label style={{font=\scriptsize}},
  legend style={{draw=none,font=\scriptsize,at={{(1.1,-0.22)}},anchor=north}},legend columns=4]
{chr(10).join(blocks)}
\end{{groupplot}}
\end{{tikzpicture}}""")


def fig_convergence():
    path = os.path.join(RES, "pipeline_V2.json")
    if not os.path.exists(path):
        print("pipeline_V2.json absent")
        return
    D = json.load(open(path))
    meth = [("random", "cmuted", "tirage aléatoire"), ("nsga2_16", "corange", "NSGA-II, $N=16$"),
            ("nsga2_8", "corange!50!black", "NSGA-II, $N=8$"), ("loop_exact", "cblue", "boucle, exact"),
            ("loop_sa", "cgreen", "boucle, recuit"), ("loop_qaoa", "cviolet", "boucle, QAOA")]
    grid = np.arange(0, 251, 2)
    blocks = []
    for sec in ("S1", "S2"):
        pl = []
        if sec == "S1":
            pl += [f"\\addlegendimage{{{c},very thick}}\\addlegendentry{{{l}}}" for k, c, l in meth if k in D[sec]]
        for k, c, l in meth:
            if k not in D[sec]:
                continue
            M = []
            for tr in D[sec][k]:
                tr = np.array(tr)
                v = []
                for g in grid:
                    j = np.searchsorted(tr[:, 0], g, side="right") - 1
                    v.append(tr[j, 1] if j >= 0 else 0.0)
                M.append(v)
            pl.append(f"\\addplot[{c},very thick,forget plot] coordinates {{{co(grid, np.median(M, axis=0))}}};")
        blocks.append(rf"""\nextgroupplot[title={{{TITLES[sec]} ({D[sec]['n_front']} profils sur le front)}}]
{chr(10).join(pl)}""")
    write("v2_convergence", rf"""\begin{{tikzpicture}}
\begin{{groupplot}}[group style={{group size=2 by 1,horizontal sep=1.4cm}},width=0.5\linewidth,height=6cm,
  xlabel={{simulations distinctes}},ylabel={{fraction du front exact trouvée}},xmin=0,xmax=250,ymin=0,ymax=1.02,
  title style={{font=\small}},tick label style={{font=\scriptsize}},
  legend style={{draw=none,font=\scriptsize,at={{(1.1,-0.25)}},anchor=north}},legend columns=3,legend cell align=left]
{chr(10).join(blocks)}
\end{{groupplot}}
\end{{tikzpicture}}""")
    NUM["summary"] = {sec: D[sec]["summary"] for sec in ("S1", "S2")}


if __name__ == "__main__":
    fig_fronts()
    fig_convergence()
    with open(os.path.join(FIG, "chiffres_v2.json"), "w") as f:
        json.dump(NUM, f, indent=1, default=float)
    print(json.dumps({k: v for k, v in NUM.items() if k != "S1" and k != "S2"}, indent=1, default=float)[:3000])
