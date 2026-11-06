"""Figures V2 du cours SER : effet du rayon de bord d'attaque (série NACA modifiée).

Usage (depuis le dossier du cours) : python3 tools/make_figures_v2.py
"""
import json
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
PROJ = os.path.abspath(os.path.join(HERE, "..", ".."))
sys.path.insert(0, os.path.join(PROJ, "Code"))

from aero.naca4 import naca4_modified, leading_edge_radius_modified  # noqa: E402
from ser.po2d import wavenumber, echo_width, radar_directions, to_db  # noqa: E402
from optim.problem_v2 import build_table_v2  # noqa: E402

FIG = os.path.join(HERE, "figures")
K = wavenumber(10e9)
NUM = {}


def write(name, body):
    with open(os.path.join(FIG, name + ".tex"), "w") as f:
        f.write(f"% figure générée par tools/make_figures_v2.py : {name}\n{body}\n")


def co(xs, ys):
    return " ".join(f"({float(x):.5g},{float(y):.5g})" for x, y in zip(xs, ys) if np.isfinite(x) and np.isfinite(y))


def mie_ratio_db(ka):
    from scipy.special import jv, hankel2
    N = int(ka + 10 * ka ** (1 / 3) + 15)
    n = np.arange(-N, N + 1)
    sm = np.sum((-1.0) ** n * jv(n, ka) / hankel2(n, ka))
    return 10 * np.log10(4 * abs(sm) ** 2 / ka / np.pi)


def fig_nez():
    Is = np.linspace(1.5, 9, 31)
    pl = []
    for t, st in ((0.08, "corange"), (0.12, "cblue"), (0.16, "cgreen")):
        po, law = [], []
        for I in Is:
            X, Y = naca4_modified(0.0, 0.4, t, I, 0.3, n_half=200)
            po.append(to_db(echo_width(X, Y, radar_directions(0.0, [0.0]), K)[0]))
            law.append(to_db(np.pi * leading_edge_radius_modified(t, I)))
        pl.append(f"\\addplot[{st},very thick] coordinates {{{co(Is, po)}}};\\addlegendentry{{PO, $t={100*t:.0f}$~\\%}}")
        pl.append(f"\\addplot[{st},dashed,forget plot] coordinates {{{co(Is, law)}}};")
        NUM[f"nez_t{100*t:.0f}"] = dict(I2=float(po[np.argmin(abs(Is - 2))]), I6=float(po[np.argmin(abs(Is - 6))]),
                                        I8=float(po[np.argmin(abs(Is - 8))]))
    write("v2_nez", rf"""\begin{{tikzpicture}}
\begin{{axis}}[width=0.72\linewidth,height=5.8cm,xlabel={{indice de rayon de bord d'attaque $I$}},ylabel={{$\sigma_{{2D}}$ de nez (dBm)}},
  xmin=1.5,xmax=9,legend style={{draw=none,font=\scriptsize}},legend pos=south east]
{chr(10).join(pl)}
\end{{axis}}
\end{{tikzpicture}}""")
    NUM["kr"] = {f"t{100*t:.0f}_I{I}": float(K * leading_edge_radius_modified(t, I)) for t in (0.08, 0.12) for I in (2, 4)}
    NUM["mie_db"] = {f"{ka:.2f}": float(mie_ratio_db(ka)) for ka in (0.16, 0.37, 0.65, 1.5)}


def fig_carte():
    T = build_table_v2()
    ts = sorted({r["t"] for r in T})
    Is = sorted({r["I"] for r in T})
    pts = []
    for t in ts:
        for I in Is:
            v = [r["SER_S1"] for r in T if r["t"] == t and r["I"] == I]
            pts.append(f"({Is.index(I)},{100*t:.0f}) [{np.median(v):.3f}]")
    write("v2_carte", rf"""\begin{{tikzpicture}}
\begin{{axis}}[width=0.5\linewidth,height=6cm,xlabel={{indice $I$}},ylabel={{épaisseur $t$ (\%)}},
  xtick={{0,1,2,3}},xticklabels={{2,4,6,8}},ytick={{6,8,...,20}},enlargelimits=false,axis on top,
  colormap={{bl}}{{color=(white) color=(cblue!45) color=(cblue!85!black)}},colorbar,
  colorbar style={{ylabel={{SER S1 (dBm)}},width=2.5mm,tick label style={{font=\scriptsize}}}},tick label style={{font=\scriptsize}}]
\addplot[matrix plot*,mesh/cols=4,point meta=explicit] coordinates {{{" ".join(pts)}}};
\end{{axis}}
\end{{tikzpicture}}""")
    S = np.array([r["SER_S1"] for r in T])
    NUM["corr_t"] = float(np.corrcoef(S, [r["t"] for r in T])[0, 1])
    NUM["corr_I"] = float(np.corrcoef(S, [r["I"] for r in T])[0, 1])
    NUM["range"] = [float(S.min()), float(S.max())]


def fig_mom():
    """Optique physique contre méthode des moments (TM), profils symétriques, secteur S1."""
    from ser.compare_po_mom import ser_both
    Is = (2.0, 3.0, 4.0, 6.0, 8.0)
    pl = []
    NUM["mom"] = {}
    for t, st in ((0.08, "corange"), (0.12, "cblue"), (0.16, "cgreen")):
        po, mo = [], []
        for I in Is:
            a, b, _ = ser_both(0.0, 0.4, t, I, 0.0, -20, 20)
            po.append(a)
            mo.append(b)
        pl.append(f"\\addplot[{st},very thick,mark=*,mark size=1.4pt] coordinates {{{co(Is, mo)}}};\\addlegendentry{{MoM, $t={100*t:.0f}$~\\%}}")
        pl.append(f"\\addplot[{st},dashed,mark=o,mark size=1.4pt,forget plot] coordinates {{{co(Is, po)}}};")
        NUM["mom"][f"t{100*t:.0f}"] = dict(I=list(Is), PO=po, MoM=mo)
    write("v2_mom", rf"""\begin{{tikzpicture}}
\begin{{axis}}[width=0.72\linewidth,height=5.8cm,xlabel={{indice de rayon de bord d'attaque $I$}},ylabel={{SER moyenne S1 (dBm)}},
  xmin=1.5,xmax=8.5,legend style={{draw=none,font=\scriptsize}},legend pos=south east]
{chr(10).join(pl)}
\end{{axis}}
\end{{tikzpicture}}""")


if __name__ == "__main__":
    fig_nez()
    fig_carte()
    fig_mom()
    with open(os.path.join(FIG, "chiffres_v2.json"), "w") as f:
        json.dump(NUM, f, indent=1)
    print(json.dumps(NUM, indent=1))
