"""Figures du cours « Contraintes et pénalités » (itération V3, lot L7).

Usage (depuis le dossier du cours) : python3 tools/make_figures.py
Lit Code/resultats/contraintes_*.json (produits par les scripts de Code/contraintes/) et écrit figures/*.tex.
"""
import json
import os

import numpy as np

HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RES = os.path.abspath(os.path.join(HERE, "..", "..", "Code", "resultats"))
FIG = os.path.join(HERE, "figures")


def co(xs, ys):
    return " ".join(f"({x:.5g},{y:.5g})" for x, y in zip(xs, ys) if y is not None and np.isfinite(y))


def write(name, body):
    with open(os.path.join(FIG, name + ".tex"), "w") as f:
        f.write("% figure générée par tools/make_figures.py\n" + body + "\n")
    print("figures/" + name + ".tex")


# ---------------------------------------------------------------- formes des pénalités
def figure_penalites(P=75.0, smax=0.5, n_slack=3, h_etoile=0.05, pente=0.8):
    h = np.linspace(-0.3, 0.5, 401)
    pas = smax / (2 ** n_slack - 1)
    s = pas * np.arange(2 ** n_slack)
    ecart = P * np.min((h[:, None] - s[None, :]) ** 2, axis=1)       # minimum sur les bits d'écart
    deseq = P * (h ** 2 - 2 * h_etoile * h)
    lag = -pente * h
    body = rf"""\begin{{tikzpicture}}
\begin{{axis}}[width=0.92\linewidth,height=6.2cm,xmin=-0.3,xmax=0.5,ymin=-0.6,ymax=2.6,
  xlabel={{marge $h=\hat C_{{L,\max}}-1{{,}}2$}},ylabel={{terme ajouté à l'énergie}},
  grid=major,grid style={{cmuted!20}},legend cell align=left,legend style={{font=\footnotesize,at={{(0.5,-0.3)}},anchor=north,legend columns=1}},
  extra x ticks={{0}},extra x tick style={{grid style={{cink,thick}}}}]
\fill[corange!8] (axis cs:-0.3,-0.6) rectangle (axis cs:0,2.6);
\node[font=\footnotesize,cink] at (axis cs:-0.15,2.35) {{violée}};
\node[font=\footnotesize,cink] at (axis cs:0.25,2.35) {{respectée}};
\addplot[cblue,very thick] coordinates {{{co(h, ecart)}}};
\addlegendentry{{bits d'écart (V2) : $P\min_s(h-s)^2$, 3 bits}}
\addplot[corange,very thick] coordinates {{{co(h, deseq)}}};
\addlegendentry{{déséquilibrée : $P(h^2-2h^\star h)$, $h^\star=0{{,}}05$}}
\addplot[cgreen,very thick] coordinates {{{co(h, lag)}}};
\addlegendentry{{lagrangien : $-\lambda h/s$, $\lambda/s=0{{,}}8$}}
\end{{axis}}
\end{{tikzpicture}}"""
    write("penalites", body)


# ---------------------------------------------------------------- seuil du classifieur
def figure_seuils():
    pv = os.path.join(RES, "contraintes_V2_seuils.json")
    pc = os.path.join(RES, "contraintes_cst.json")
    if not (os.path.exists(pv) and os.path.exists(pc)):
        return
    V = json.load(open(pv))
    C = json.load(open(pc))["convergence"]
    S = V["seuils"]
    sc = sorted(float(k) for k in C["seuils"])
    ev = [100 * C["seuils"][str(k)]["echecs_evites"] for k in sc]
    sa = [100 * C["seuils"][str(k)]["succes_sacrifies"] for k in sc]
    fr = C["front_proba_tout_cache"]
    succ = [100 * V["S2"][str(k)]["succes"] for k in S]
    pf = [100 * V["S2"][str(k)]["part_faisable"] for k in S]
    marks = " ".join(f"({p:.3f},96)" for p in fr)
    body = rf"""\begin{{tikzpicture}}
\begin{{groupplot}}[group style={{group size=2 by 1,horizontal sep=1.7cm}},width=0.49\linewidth,height=5.6cm,
  xmin=0,xmax=0.55,ymin=0,ymax=105,xlabel={{seuil $p_{{\min}}$}},grid=major,grid style={{cmuted!20}},
  legend style={{font=\scriptsize,at={{(0.5,1.03)}},anchor=south,legend columns=1}},legend cell align=left]
\nextgroupplot[title={{\small CST, rejeu de la campagne}},ylabel={{\%}}]
\addplot[cgreen,very thick,mark=*] coordinates {{{co(sc, ev)}}};
\addlegendentry{{échecs d'XFOIL évités}}
\addplot[corange,very thick,mark=square*] coordinates {{{co(sc, sa)}}};
\addlegendentry{{succès sacrifiés}}
\addplot[cink,only marks,mark=|,mark size=6pt,very thick] coordinates {{{marks}}};
\addlegendentry{{$P$(succès) des 10 profils du front}}
\nextgroupplot[title={{\small V2, secteur S2, 20 graines}}]
\addplot[cblue,very thick,mark=*] coordinates {{{co(S, succ)}}};
\addlegendentry{{front exact retrouvé (\% des graines)}}
\addplot[cmuted,very thick,mark=triangle*] coordinates {{{co(S, pf)}}};
\addlegendentry{{profils simulés faisables}}
\end{{groupplot}}
\end{{tikzpicture}}"""
    write("seuils", body)


# ---------------------------------------------------------------- campagne CST
def figure_campagne():
    p = os.path.join(RES, "contraintes_cst_campagne.json")
    if not os.path.exists(p):
        return
    D = json.load(open(p))
    st = {"hasard": ("cmuted,thick,dashed", "hasard"), "nsga2_16": ("cink,thick,dotted", "NSGA-II"),
          "quadz_modele": ("cblue,very thick,mark=*", "bits d'écart (actuel)"),
          "fm_modele": ("corange,thick,mark=square*", "machine de factorisation (bits d'écart)"),
          "quadz_lagrange": ("cgreen,very thick,mark=triangle*", "lagrangien vérifié"),
          "quadz_lagrange_clf": ("cviolet,very thick,mark=diamond*", "lagrangien + classifieur")}
    lines = []
    for m, (s, lab) in st.items():
        if m in D:
            lines.append(rf"\addplot[{s},mark size=1.4pt] coordinates {{{co(D[m]['grille'], D[m]['hv_median'])}}};"
                         + "\n" + rf"\addlegendentry{{{lab}}}")
    body = r"""\begin{tikzpicture}
\begin{axis}[width=0.92\linewidth,height=7cm,xmin=0,xmax=300,ymin=0.3,ymax=1,xlabel={simulations},
  ylabel={$HV/HV_{\rm ref}$ (médiane sur 8 graines)},grid=major,grid style={cmuted!20},legend pos=south east,
  legend cell align=left,legend style={font=\footnotesize}]
""" + "\n".join(lines) + r"""
\end{axis}
\end{tikzpicture}"""
    write("campagne_hv", body)


if __name__ == "__main__":
    os.makedirs(FIG, exist_ok=True)
    figure_penalites()
    figure_seuils()
    figure_campagne()
