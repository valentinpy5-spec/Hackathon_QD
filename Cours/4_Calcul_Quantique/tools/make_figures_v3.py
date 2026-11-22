"""Figures V3 du cours quantique : anatomie des QUBO de la boucle (lit Code/resultats/anatomie_qubo_V3.json).

Usage (depuis le dossier du cours) : python3 tools/make_figures_v3.py
Produit figures/v3_anatomie.tex et affiche les chiffres cités dans le chapitre.
"""
import json
import os

import numpy as np

HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
PROJ = os.path.abspath(os.path.join(HERE, "..", ".."))
D = json.load(open(os.path.join(PROJ, "Code", "resultats", "anatomie_qubo_V3.json")))
FIG = os.path.join(HERE, "figures")


def co(xs, ys):
    return " ".join(f"({x:.5g},{y:.5g})" for x, y in zip(xs, ys))


# ---------------------------------------------------------------- spectre (S2, w = 0,5, début de boucle)
sp = D["spectre"]
rang = np.array(sp["rang"]) + 1
E = np.array(sp["E_rel"])
share = sp["obj_share"]
n_below = int(np.interp(share, E, rang))

# ---------------------------------------------------------------- QAOA : gain sur le hasard, avec et sans contraintes
gain = {"complet": [], "sans_contraintes": []}
for var in gain:
    for p_i in range(3):
        g = [v[var][p_i]["p_opt"] / v[var][p_i]["p_random"] for v in D["qaoa"].values()]
        gain[var].append((float(np.median(g)), float(np.min(g)), float(np.max(g))))
P = (1, 2, 3)

body = rf"""% figure générée par tools/make_figures_v3.py
\begin{{tikzpicture}}
\begin{{groupplot}}[group style={{group size=2 by 1,horizontal sep=1.7cm}},width=0.48\linewidth,height=5.8cm,
  title style={{font=\small}},tick label style={{font=\scriptsize}},label style={{font=\small}},
  legend style={{draw=none,font=\scriptsize,fill=none}}]
\nextgroupplot[xmode=log,ymode=log,xlabel={{rang de l'état (énergie croissante)}},ylabel={{$(E-E_{{\min}})/(E_{{\max}}-E_{{\min}})$}},
  title={{spectre d'un QUBO de la boucle}},ymin=1e-4,ymax=1.5,xmin=2,xmax=16384,legend pos=north west]
\addplot[cviolet,very thick] coordinates {{{co(rang[1:], E[1:])}}};\addlegendentry{{les $2^{{14}}$ états}}
\addplot[corange,thick,dashed,domain=2:16384,samples=2] {{{share:.4f}}};\addlegendentry{{étendue de l'objectif ({100 * share:.1f}~\%)}}
\nextgroupplot[ymode=log,xlabel={{profondeur $p$}},ylabel={{$P(\text{{opt}})\,/\,P_{{\text{{hasard}}}}$}},xtick={{1,2,3}},
  title={{QAOA : gain sur le hasard}},xmin=0.7,xmax=3.3,ymin=0.5,ymax=1000,legend style={{at={{(0.5,-0.3)}},anchor=north}}]
\addplot[cviolet,very thick,mark=*,mark size=1.4pt,error bars/.cd,y dir=both,y explicit]
  coordinates {{{" ".join(f"({p},{m:.4g}) -= (0,{m - lo:.4g}) += (0,{hi - m:.4g})" for p, (m, lo, hi) in zip(P, gain["complet"]))}}};
\addlegendentry{{QUBO de la boucle (14 qubits)}}
\addplot[cgreen,very thick,mark=square*,mark size=1.4pt,error bars/.cd,y dir=both,y explicit]
  coordinates {{{" ".join(f"({p},{m:.4g}) -= (0,{m - lo:.4g}) += (0,{hi - m:.4g})" for p, (m, lo, hi) in zip(P, gain["sans_contraintes"]))}}};
\addlegendentry{{même objectif, sans contraintes (10 qubits)}}
\end{{groupplot}}
\end{{tikzpicture}}
"""
open(os.path.join(FIG, "v3_anatomie.tex"), "w").write(body)

# ---------------------------------------------------------------- chiffres du chapitre
print(f"part de l'objectif (spectre) {share:.4f} ; états sous cette part : {n_below}")
for var in gain:
    print(var, [(round(m, 1), round(lo, 1), round(hi, 1)) for m, lo, hi in gain[var]])
for k, v in D["qaoa"].items():
    print(k, "P(opt) p=1..3", [f"{q['p_opt']:.1e}" for q in v["complet"]],
          "vu en 512 tirs", [f"{q['p_seen_512']:.2f}" for q in v["complet"]],
          "| sans contraintes", [f"{q['p_opt']:.3f}" for q in v["sans_contraintes"]],
          "s", [round(q["seconds"]) for q in v["complet"]])
for S, res in D["anatomy"].items():
    for name, r in res.items():
        f = lambda key: np.median([x[key] for x in r])
        print(S, name, f"objectif {100 * f('obj_share'):.1f} % (min {100 * min(x['obj_share'] for x in r):.1f}, "
              f"max {100 * max(x['obj_share'] for x in r):.1f}), écart {f('gap'):.1e}, |Q| pén./obj. {np.median([x['Q_max_pen'] / x['Q_max_obj'] for x in r]):.0f}, "
              f"= substitution {np.mean([x['same_as_substitute'] for x in r]):.0%}, prédit faisable "
              f"{np.mean([x['qubo_pred_feasible'] for x in r]):.0%}, énum {1e3 * f('t_enum'):.1f} ms, n={len(r)}")
print("XFOIL s/profil", D["xfoil_s_par_profil"])
