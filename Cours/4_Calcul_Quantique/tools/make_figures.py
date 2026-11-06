"""Génère les figures pgfplots « à données » du cours depuis le JSON des expériences.

Source des données : Code/resultats/experiences_quantique.json
(produit par `python3 -m quantique.experiences` dans Code/). Les coordonnées
sont embarquées dans les fichiers : le .tex se compile sans Python.

Usage (depuis LaTeX/) : python3 tools/make_figures.py
"""
import json
import os

import numpy as np

HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
PROJ = os.path.abspath(os.path.join(HERE, "..", ".."))   # racine du projet HackathonQuantique
SRC = os.path.join(PROJ, "Code", "resultats",
                   "experiences_quantique.json")
FIG = os.path.join(HERE, "figures")
os.makedirs(FIG, exist_ok=True)
D = json.load(open(SRC))
T, A, L = D["toy"], D["aero"], D["boucle"]


def write(name, body):
    with open(os.path.join(FIG, name + ".tex"), "w") as f:
        f.write(f"% figure générée par tools/make_figures.py : {name}\n{body}\n")


def coords(xs, ys, fmt="{:.4g}"):
    return " ".join(f"({fmt.format(x)},{fmt.format(y)})" for x, y in zip(xs, ys))


# ------------------------------------------------------------ spectre adiabatique
s = np.array(T["s"])
spec = np.array(T["spectrum"])
plots = []
for k in range(spec.shape[1]):
    style = "cblue,very thick" if k == 0 else ("corange,thick" if k == 1 else "cmuted,thin")
    plots.append(f"\\addplot[{style}] coordinates {{{coords(s[::2], spec[::2, k])}}};")
i = int(np.argmin(spec[:, 1] - spec[:, 0]))
write("spectre_adiabatique", rf"""\begin{{tikzpicture}}
\begin{{axis}}[width=0.8\linewidth,height=6.2cm,axis lines=left,xmin=0,xmax=1,ymin=-3.3,ymax=1.3,
  xlabel={{$s=t/T$}},ylabel={{niveaux d'énergie de $H(s)$}},
  legend style={{draw=none,font=\small,at={{(0.02,0.02)}},anchor=south west,fill=white,fill opacity=0.85,text opacity=1}}]
{chr(10).join(plots)}
\draw[cink,<->,thick] (axis cs:{s[i]:.3f},{spec[i,0]:.3f}) -- (axis cs:{s[i]:.3f},{spec[i,1]:.3f})
  node[midway,right,font=\small,fill=white] {{$g_{{\min}}={T["gap_min"]:.2f}$}};
\node[font=\small,anchor=west] at (axis cs:0.02,-2.75) {{$H_B=-\sum_k X_k$ : fondamental $|{{+}}{{+}}{{+}}\rangle$}};
\node[font=\small,anchor=east] at (axis cs:0.98,-2.25) {{$H_C$ : fondamental $|101\rangle$, $E=-2$}};
\end{{axis}}
\end{{tikzpicture}}""")

# ------------------------------------------------------------ paysage QAOA p = 1 (jouet)
g = np.array(T["land_gamma"])
b = np.array(T["land_beta"])
Z = np.array(T["landscape"])
rows = []
for jb, bb in enumerate(b):
    rows.append(" ".join(f"({gg:.4f},{bb:.4f},{Z[jb, ig]:.4f})" for ig, gg in enumerate(g)))
jb, ig = np.unravel_index(np.argmin(Z), Z.shape)
write("paysage_qaoa_p1", rf"""\begin{{tikzpicture}}
\begin{{axis}}[width=0.78\linewidth,height=6cm,view={{0}}{{90}},xmin=0,xmax=6.2832,ymin=0,ymax=3.1416,
  xlabel={{$\gamma$}},ylabel={{$\beta$}},xtick={{0,1.5708,3.1416,4.7124,6.2832}},
  xticklabels={{$0$,$\pi/2$,$\pi$,$3\pi/2$,$2\pi$}},ytick={{0,1.5708,3.1416}},yticklabels={{$0$,$\pi/2$,$\pi$}},
  colormap={{seq}}{{color=(cblue!85!black) color=(cblue!45) color=(white)}},
  colorbar,colorbar style={{ylabel={{$F_1(\gamma,\beta)$}}}},axis on top,enlargelimits=false]
\addplot3[surf,shader=interp,mesh/rows={len(b)}] coordinates {{
{chr(10).join(rows)}
}};
\addplot3[only marks,mark=x,mark size=4pt,corange,very thick] coordinates {{({g[ig]:.4f},{b[jb]:.4f},0)}};
\end{{axis}}
\end{{tikzpicture}}""")
T["land_min"] = dict(gamma=float(g[ig]), beta=float(b[jb]), F=float(Z[jb, ig]))

# ------------------------------------------------------------ distribution QAOA jouet p = 1
pr = np.array(T["p1_prob"])
labels = ["".join(str(x) for x in bits) for bits in T["bits"]]
E = np.array(T["E"])
bars = coords(range(8), pr)
write("jouet_proba_p1", rf"""\begin{{tikzpicture}}
\begin{{axis}}[width=0.8\linewidth,height=5cm,ybar,bar width=12pt,axis lines=left,ymin=0,ymax=0.4,
  xtick={{0,...,7}},xticklabels={{{",".join(f"$|{l}\\rangle$" for l in labels)}}},
  ylabel={{probabilité}},xticklabel style={{font=\small}},
  nodes near coords={{}},enlarge x limits=0.08]
\addplot[fill=cblue,draw=none] coordinates {{{bars}}};
\addplot[corange,thick,sharp plot,mark=none,update limits=false] coordinates {{(-0.5,0.125) (7.5,0.125)}};
\end{{axis}}
\end{{tikzpicture}}""")

# ------------------------------------------------------------ QAOA aéro : proba par rang
cols = {1: "cblue", 3: "corange", 5: "cgreen"}
pl = []
for q in A["qaoa"]:
    if q["p"] in cols:
        pl.append(f"\\addplot[{cols[q['p']]},thick,mark=*,mark size=1.6pt] coordinates "
                  f"{{{coords(range(1, 26), q['prob_by_rank'])}}};\\addlegendentry{{$p={q['p']}$}}")
write("qaoa_proba_rang", rf"""\begin{{tikzpicture}}
\begin{{axis}}[width=0.85\linewidth,height=5.8cm,axis lines=left,xmin=0.5,xmax=25.5,ymin=0,ymax=0.155,
  xlabel={{rang de l'énergie QUBO (1 = minimum)}},ylabel={{probabilité de mesure}},
  legend style={{draw=none,font=\small,at={{(0.98,0.98)}},anchor=north east}}]
{chr(10).join(pl)}
\addplot[cmuted,dashed,thick] coordinates {{(0.5,{1/512:.5f}) (25.5,{1/512:.5f})}};\addlegendentry{{hasard : $1/512$}}
\end{{axis}}
\end{{tikzpicture}}""")

# ------------------------------------------------------------ QAOA aéro : convergence en p
ps = [q["p"] for q in A["qaoa"]]
write("qaoa_convergence_p", rf"""\begin{{tikzpicture}}
\begin{{axis}}[width=0.7\linewidth,height=5.4cm,axis lines=left,xmin=0.7,xmax=5.3,ymin=0,ymax=1,xtick={{1,...,5}},
  xlabel={{profondeur $p$}},ylabel={{probabilité / ratio}},
  legend style={{draw=none,font=\small,at={{(0.02,0.62)}},anchor=west}}]
\addplot[cblue,thick,mark=*] coordinates {{{coords(ps, [q['prob_opt'] for q in A['qaoa']])}}};\addlegendentry{{$P(\text{{optimum}})$}}
\addplot[corange,thick,mark=square*] coordinates {{{coords(ps, [q['prob_top5'] for q in A['qaoa']])}}};\addlegendentry{{$P(\text{{5 meilleurs}})$}}
\addplot[cgreen,thick,mark=triangle*] coordinates {{{coords(ps, [q['ratio'] for q in A['qaoa']])}}};\addlegendentry{{ratio d'approximation $r$}}
\end{{axis}}
\end{{tikzpicture}}""")

# ------------------------------------------------------------ angles optimaux p = 5
q5 = A["qaoa"][4]
k = np.arange(1, 6)
write("qaoa_angles", rf"""\begin{{tikzpicture}}
\begin{{axis}}[width=0.7\linewidth,height=5.2cm,axis lines=left,xmin=0.7,xmax=5.3,xtick={{1,...,5}},
  xlabel={{couche $k$}},ylabel={{angle (rad)}},
  legend style={{draw=none,font=\small,at={{(0.02,0.98)}},anchor=north west}}]
\addplot[cblue,thick,mark=*] coordinates {{{coords(k, q5['gammas'])}}};\addlegendentry{{$\gamma_k$ (coût)}}
\addplot[corange,thick,mark=square*] coordinates {{{coords(k, np.abs(q5['betas']))}}};\addlegendentry{{$|\beta_k|$ (mélangeur)}}
\end{{axis}}
\end{{tikzpicture}}""")

# ------------------------------------------------------------ recuit simulé
sw = [x["sweeps"] for x in A["sa"]]
write("recuit_succes", rf"""\begin{{tikzpicture}}
\begin{{semilogxaxis}}[width=0.7\linewidth,height=5.2cm,axis lines=left,ymin=0,ymax=1.02,
  xlabel={{nombre de balayages (pas de température)}},ylabel={{fraction des 200 lectures}},
  legend style={{draw=none,font=\small,at={{(0.98,0.03)}},anchor=south east}}]
\addplot[cblue,thick,mark=*] coordinates {{{coords(sw, [x['success'] for x in A['sa']])}}};\addlegendentry{{minimum exact du QUBO}}
\addplot[corange,thick,mark=square*] coordinates {{{coords(sw, [x['success_01'] for x in A['sa']])}}};\addlegendentry{{à moins de 0{{,}}1 count du minimum}}
\end{{semilogxaxis}}
\end{{tikzpicture}}""")

# ------------------------------------------------------------ boucle itérative
mc = np.array(L["median_curve"])
rc = np.array(L["random_curve"])
n0 = L["n_init"]
xs_loop = n0 + np.arange(len(mc))          # une valeur par simulation (voir experiences.boucle)
xs_rand = n0 + np.arange(len(rc))
write("boucle_convergence", rf"""\begin{{tikzpicture}}
\begin{{axis}}[width=0.85\linewidth,height=5.6cm,axis lines=left,xmin={n0},xmax=60,ymin=49,ymax=62,
  xlabel={{nombre de profils simulés}},ylabel={{meilleur $C_D$ trouvé (counts), médiane}},
  legend style={{draw=none,font=\small,at={{(0.98,0.98)}},anchor=north east}}]
\addplot[cblue,very thick] coordinates {{{coords(xs_loop[xs_loop <= 60], mc[xs_loop <= 60])}}};\addlegendentry{{boucle : modèle quadratique, puis QUBO}}
\addplot[corange,thick] coordinates {{{coords(xs_rand, rc)}}};\addlegendentry{{tirage aléatoire}}
\addplot[cmuted,dashed] coordinates {{({n0},{L['true_min']:.3f}) (60,{L['true_min']:.3f})}};\addlegendentry{{vrai minimum ({L['true_min']:.1f})}}
\end{{axis}}
\end{{tikzpicture}}""")

# ------------------------------------------------------------ matrice Q du QUBO aéro
# Q recalculée ici depuis le JSON n'est pas stockée : on la reconstruit à partir du code
import sys
sys.path.insert(0, os.path.join(PROJ, "Code"))
from quantique.experiences import load_base  # noqa: E402
from quantique.qubo import quadratic_features, surrogate_to_qubo  # noqa: E402

I, CD, _ = load_base()
coef, *_ = np.linalg.lstsq(quadratic_features(I / 7 * 2 - 1), CD, rcond=None)
Q, c = surrogate_to_qubo(coef, [3, 3, 3], [8, 8, 8])
vmax = np.abs(Q).max()
cells = []
names = ["m_0", "m_1", "m_2", "p_0", "p_1", "p_2", "t_0", "t_1", "t_2"]
for i in range(9):
    for j in range(9):
        if j < i:
            continue
        v = Q[i, j] / vmax
        col = f"cblue!{abs(v)*85:.0f}" if v < 0 else f"corange!{abs(v)*85:.0f}"
        cells.append(f"\\fill[{col}] ({j},{-i}) rectangle ++(1,-1);")
        cells.append(f"\\node[font=\\tiny] at ({j+0.5},{-i-0.5}) {{{Q[i, j]:.1f}}};")
labs = [f"\\node[font=\\scriptsize] at ({j+0.5},0.35) {{${names[j]}$}};" for j in range(9)]
labs += [f"\\node[font=\\scriptsize] at (-0.45,{-i-0.5}) {{${names[i]}$}};" for i in range(9)]
write("matrice_Q", rf"""\begin{{tikzpicture}}[scale=0.62]
{chr(10).join(cells)}
\draw[cmuted,thin] (0,0) grid (9,-9);
\draw[cink,thick] (0,0) rectangle (3,-3) (3,-3) rectangle (6,-6) (6,-6) rectangle (9,-9);
{chr(10).join(labs)}
\end{{tikzpicture}}""")

# ------------------------------------------------------------ valeurs pour le texte
vals = dict(gap_min=T["gap_min"], s_gap=T["s_gap_min"], toy_p1_P=T["p1_prob"][T["opt_index"]],
            toy_p2_P=T["p2_prob_opt"], toy_p3_P=T["p3_prob_opt"], toy_p1_E=T["p1_energy"],
            land_min=T["land_min"], qubo_const=c, Qmax=vmax,
            aero={k: A[k] for k in ("qubo_min", "qubo_second", "qubo_argmin_true_rank",
                                    "true_min", "true_at_qubo_min", "raw_p1_prob_opt",
                                    "n_nonzero_offdiag")},
            qaoa=[{k: q[k] for k in ("p", "prob_opt", "prob_top5", "ratio", "shots_to_opt")}
                  for q in A["qaoa"]],
            gammas5=q5["gammas"], betas5=q5["betas"],
            boucle={k: L[k] for k in ("median_sims", "q25", "q75", "random_expected")})
print(json.dumps(vals, indent=1))
