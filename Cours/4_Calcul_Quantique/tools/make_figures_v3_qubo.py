"""Figures V3 du cours quantique : le QUBO de la V2 et la boucle racontée sur une exécution réelle.

Lit Code/resultats/trace_boucle_V2.json (python3 -m quantique.trace_boucle_v2) et produit :
    figures/matrice_Q_v2.tex      matrice Q à 14 qubits du premier QUBO (S2, w = 0,5)
    figures/penalite_cl.tex       pénalité effective de manœuvre en fonction de C_L,max prédit
    figures/boucle_trace_v2.tex   front retrouvé et candidats faisables au fil des itérations
Usage (depuis le dossier du cours) : python3 tools/make_figures_v3_qubo.py
"""
import json
import os

import numpy as np

HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
PROJ = os.path.abspath(os.path.join(HERE, "..", ".."))
D = json.load(open(os.path.join(PROJ, "Code", "resultats", "trace_boucle_V2.json")))
FIG = os.path.join(HERE, "figures")
q = D["qubo_w05"]


def write(name, body):
    with open(os.path.join(FIG, name + ".tex"), "w") as fh:
        fh.write("% figure générée par tools/make_figures_v3_qubo.py\n" + body + "\n")


# ---------------------------------------------------------------- matrice Q (échelle logarithmique)
Q = np.array(q["Q"])
n = Q.shape[0]
lo, hi = 1e-2, np.abs(Q).max()
labels = ["m_0", "m_1", "m_2", "p_0", "p_1", "t_0", "t_1", "t_2", "I_0", "I_1", "y", "s_0", "s_1", "s_2"]
cells = []
for i in range(n):
    for j in range(i, n):
        v = Q[i, j]
        if abs(v) < 1e-12:
            continue
        k = np.clip(np.log10(abs(v) / lo) / np.log10(hi / lo), 0.05, 1.0)
        col = "corange" if v > 0 else "cblue"
        cells.append(rf"\fill[{col}!{100 * k:.0f}] ({j},{-i}) rectangle ++(1,-1);")
labs = [rf"\node[font=\scriptsize,anchor=east] at (-0.05,{-i - 0.5}) {{${l}$}};" for i, l in enumerate(labels)]
labs += [rf"\node[font=\scriptsize,anchor=south] at ({i + 0.5},0.05) {{${l}$}};" for i, l in enumerate(labels)]
write("matrice_Q_v2", rf"""\begin{{tikzpicture}}[scale=0.5]
{chr(10).join(cells)}
\draw[cmuted,very thin] (0,0) grid ({n},{-n});
\draw[cink,thick] (0,0) rectangle (10,-10) (10,-10) rectangle (11,-11) (11,-11) rectangle (14,-14);
{chr(10).join(labs)}
\node[font=\small,align=left,anchor=west] at (14.6,-2.5) {{bits de conception\\ $(m,p,t,I)$ : 10 qubits}};
\node[font=\small,align=left,anchor=west] at (14.6,-10.5) {{auxiliaire $y$ (épaisseur)}};
\node[font=\small,align=left,anchor=west] at (14.6,-12.5) {{écart $s$ (manœuvre)}};
\fill[cblue] (14.6,-6.2) rectangle ++(0.6,-0.6); \node[font=\scriptsize,anchor=west] at (15.3,-6.5) {{négatif}};
\fill[corange] (14.6,-7.2) rectangle ++(0.6,-0.6); \node[font=\scriptsize,anchor=west] at (15.3,-7.5) {{positif}};
\node[font=\scriptsize,anchor=west,align=left] at (14.6,-8.6) {{teinte $\propto\log_{{10}}|Q_{{ij}}|$,\\ de $10^{{-2}}$ à {hi:.0f}}};
\end{{tikzpicture}}""")

# ---------------------------------------------------------------- pénalité effective de manœuvre
L0, a = q["L_const"], np.array(q["L_coefs"])
Lmax = L0 + a[a > 0].sum()
smax = Lmax - 1.2
step = smax / 7
W = 3.0 * D["cons"].get("cl_weight", 25.0)
Ls = np.linspace(1.0, 1.75, 601)
eff = [min(W * (L - 1.2 - k * step) ** 2 for k in range(8)) for L in Ls]
ideal = [W * max(0.0, 1.2 - L) ** 2 for L in Ls]
co = lambda xs, ys: " ".join(f"({x:.4f},{y:.4f})" for x, y in zip(xs, ys))
write("penalite_cl", rf"""\begin{{tikzpicture}}
\begin{{axis}}[width=0.85\linewidth,height=5.6cm,xlabel={{$\widehat{{C_{{L,\max}}}}$ prédit par le modèle affine}},ylabel={{pénalité}},
  xmin=1.0,xmax=1.75,ymin=0,ymax=0.6,tick label style={{font=\scriptsize}},legend style={{draw=none,font=\scriptsize,at={{(0.98,0.97)}},anchor=north east}}]
\addplot[cviolet,very thick] coordinates {{{co(Ls, eff)}}};\addlegendentry{{pénalité du QUBO (écart sur 3 bits, minimum sur $s$)}}
\addplot[cmuted,dashed,thick] coordinates {{{co(Ls, ideal)}}};\addlegendentry{{pénalité idéale (écart continu)}}
\draw[corange,thick] (axis cs:1.2,0) -- (axis cs:1.2,0.6) node[pos=0.93,left,font=\scriptsize] {{seuil 1,2}};
\addplot[only marks,mark=*,mark size=1.8pt,cgreen] coordinates {{({q['L_argmin']:.4f},{W * (1.2 - q['L_argmin']) ** 2:.4f})}};
\node[cgreen,font=\scriptsize,anchor=west,align=left] (lab) at (axis cs:1.005,0.36) {{minimum du QUBO\\ ($\widehat{{C_{{L,\max}}}}={q['L_argmin']:.3f}$)}};
\draw[cgreen,->,>=stealth] (lab.south) -- (axis cs:{q['L_argmin'] - 0.004:.3f},{W * (1.2 - q['L_argmin']) ** 2 + 0.015:.3f});
\end{{axis}}
\end{{tikzpicture}}""")

# ---------------------------------------------------------------- la boucle au fil des itérations
its = D["iterations"]
sims = [D["iterations"][0]["n_sims"]] + [it["n_sims"] + len(it["cand"]) for it in its]
found = [0] + [it["front_found"] for it in its]
feas = [sum(c["feasible"] for c in it["cand"]) for it in its]
x_it = [it["n_sims"] + len(it["cand"]) for it in its]
write("boucle_trace_v2", rf"""\begin{{tikzpicture}}
\begin{{groupplot}}[group style={{group size=1 by 2,vertical sep=0.9cm,xticklabels at=edge bottom}},width=0.9\linewidth,
  xmin=8,xmax={sims[-1] + 5},tick label style={{font=\scriptsize}},label style={{font=\small}}]
\nextgroupplot[height=4.8cm,ylabel={{profils du front}},ymin=0,ymax=12,ytick={{0,2,4,6,8,10}}]
\addplot[cblue,very thick,const plot] coordinates {{{co(sims, found)}}};
\draw[cmuted,dashed] (axis cs:8,10) -- (axis cs:{sims[-1] + 5},10);
\draw[cmuted,dotted] (axis cs:30,0) -- (axis cs:30,12) (axis cs:100,0) -- (axis cs:100,12);
\node[font=\scriptsize] at (axis cs:20,11) {{phase 1}};
\node[font=\scriptsize] at (axis cs:65,11) {{phase 2}};
\node[font=\scriptsize] at (axis cs:142,11) {{phase 3}};
\nextgroupplot[height=3.6cm,xlabel={{nombre de profils simulés}},ylabel={{candidats faisables}},ymin=0,ymax=5.5,ytick={{0,1,2,3,4,5}},ybar,bar width=2.2pt]
\addplot[fill=cgreen!70,draw=none] coordinates {{{co(x_it, feas)}}};
\end{{groupplot}}
\end{{tikzpicture}}""")

print(f"L0={L0:.3f}, Lmax={Lmax:.3f}, s_max={smax:.3f}, pas={step:.4f}, pénalité résiduelle max={W * (step / 2) ** 2:.3f}")
print("itérations :", len(its), "simulations finales :", sims[-1])
