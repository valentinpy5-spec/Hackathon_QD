"""Figures et chiffres du cours « IA et QUBO » (itération V3, lot L6).

Usage (depuis le dossier du cours) : python3 tools/make_figures.py
Lit Code/resultats/ia_qubo_V2.json, ia_qubo_precision.json, ia_qubo_echelle.json (s'ils existent)
et écrit figures/*.tex et figures/chiffres.json.
"""
import json
import os

import numpy as np

HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RES = os.path.abspath(os.path.join(HERE, "..", "..", "Code", "resultats"))
FIG = os.path.join(HERE, "figures")
CH = {}

NOMS = {"quadz_modele": "quadratique en $z$ (actuel)", "fm_modele": "machine de factorisation",
        "bocs_modele": "bayésien sur les bits", "mlp_modele": "réseau de neurones",
        "quadz_bn": "quadratique en $z$, boîte noire", "fm_bn": "factorisation, boîte noire",
        "bocs_bn": "bayésien, boîte noire", "mlp_bn": "réseau, boîte noire",
        "hasard": "hasard", "nsga2_16": "NSGA-II"}
STYLE = {"quadz": "cblue,very thick,mark=*", "fm": "corange,very thick,mark=square*",
         "bocs": "cgreen,very thick,mark=triangle*", "mlp": "cviolet,very thick,mark=diamond*",
         "hasard": "cmuted,thick,dashed", "nsga2": "cink,thick,dotted"}


def style(m):
    s = STYLE[m.split("_")[0]]
    return s + (",dashed" if m.endswith("_bn") else "")


def co(xs, ys):
    return " ".join(f"({x:.5g},{y:.5g})" for x, y in zip(xs, ys) if y is not None and np.isfinite(y))


def write(name, body):
    with open(os.path.join(FIG, name + ".tex"), "w") as f:
        f.write("% figure générée par tools/make_figures.py\n" + body + "\n")


# ---------------------------------------------------------------- précision des modèles
def figures_precision():
    p = os.path.join(RES, "ia_qubo_precision.json")
    if not os.path.exists(p):
        return
    D = json.load(open(p))
    N = D["tailles"]
    for jeu in ("v2", "cst"):
        if jeu not in D:
            continue
        plots = []
        for cible, titre in (("CD_counts", "traînée $C_D$"), ("SER_db", "SER"), ("CLmax", "$C_{L,\\max}$")):
            for crit, lab in (("err", "erreur relative"), ("rho", "corrélation de rang")):
                lines = []
                for m in ("quadz", "fm", "bocs", "mlp"):
                    v = D[jeu][cible][m][crit]
                    lines.append(rf"\addplot[{STYLE[m]},mark size=1.2pt] coordinates {{{co(N, v)}}};")
                ymin, ymax = (0, 1.2) if crit == "err" else (0, 1)
                plots.append(rf"\nextgroupplot[title={{{titre}}},ylabel={{{lab}}},ymin={ymin},ymax={ymax}]" + "\n" + "\n".join(lines))
            CH[f"prec_{jeu}_{cible}"] = {m: D[jeu][cible][m] for m in ("quadz", "fm", "bocs", "mlp")}
        # deux lignes : erreur (haut) et rang (bas), trois colonnes
        order = [plots[0], plots[2], plots[4], plots[1], plots[3], plots[5]]
        legend = (r"\legend{quadratique en $z$, machine de factorisation, bayésien sur les bits, réseau de neurones}")
        write(f"precision_{jeu}", rf"""\begin{{tikzpicture}}
\begin{{groupplot}}[group style={{group size=3 by 2,horizontal sep=1.15cm,vertical sep=2.3cm}},width=0.33\linewidth,height=4.6cm,
  xmode=log,xlabel={{profils d'entraînement $N$}},tick label style={{font=\scriptsize}},label style={{font=\scriptsize}},
  title style={{font=\small}},legend style={{font=\scriptsize,draw=none,at={{(-0.75,-0.45)}},anchor=north,legend columns=4}}]
{chr(10).join(order[:5])}
{order[5]}
{legend}
\end{{groupplot}}
\end{{tikzpicture}}""")


# ---------------------------------------------------------------- résultats V2
def figures_v2():
    p = os.path.join(RES, "ia_qubo_V2.json")
    if not os.path.exists(p):
        return
    D = json.load(open(p))
    meth = [m for m in NOMS if m in D["S1"]]
    grille = list(range(10, 251, 10))
    CH["v2"] = {}
    for sec in ("S1", "S2"):
        CH["v2"][sec] = {}
        lines = []
        for m in meth:
            traces = D[sec][m]
            M = []
            for tr in traces:
                row = []
                for k in grille:
                    vals = [r[1] for r in tr if r[0] <= k]
                    row.append(vals[-1] if vals else 0.0)
                M.append(row)
            med = np.median(np.array(M), axis=0)
            lines.append(rf"\addplot[{style(m)},mark size=1pt,mark repeat=3] coordinates {{{co(grille, med)}}};")
            s = D[sec]["summary"][m]
            fin = [tr[-1] for tr in traces]
            CH["v2"][sec][m] = dict(front=s["full_front"], succes=s["success"], hv99=s["hv99"],
                                    part_finale=float(np.median([f[1] for f in fin])),
                                    secondes=float(np.mean(D[sec][m + "_seconds"])))
        leg = ", ".join("{" + NOMS[m] + "}" for m in meth)
        write(f"v2_front_{sec}", rf"""\begin{{tikzpicture}}
\begin{{axis}}[width=0.95\linewidth,height=6.2cm,xlabel={{profils simulés}},ylabel={{part du front exact retrouvée (médiane)}},
  xmin=10,xmax=250,ymin=0,ymax=1.02,tick label style={{font=\scriptsize}},label style={{font=\small}},
  legend style={{font=\scriptsize,draw=none,at={{(0.5,-0.22)}},anchor=north,legend columns=2}},legend cell align=left]
{chr(10).join(lines)}
\legend{{{leg}}}
\end{{axis}}
\end{{tikzpicture}}""")


# ---------------------------------------------------------------- résultats à l'échelle
def figures_echelle():
    p = os.path.join(RES, "ia_qubo_echelle.json")
    if not os.path.exists(p):
        return
    D = json.load(open(p))
    meth = [m for m in NOMS if m in D]
    lines = []
    CH["echelle"] = dict(n_simules_total=D["n_simules_total"], n_faisables_total=D["n_faisables_total"],
                         n_front_ref=len(D["front_ref"]), duree_min=D["duree_s"] / 60)
    for m in meth:
        r = D[m]
        lines.append(rf"\addplot[{style(m)},mark size=1pt] coordinates {{{co(r['grille'], r['hv_median'])}}};")
        CH["echelle"][m] = dict(hv100=r["hv_median"][4], hv200=r["hv_median"][9], hv300=r["hv_median"][-1],
                                hv_final=r["hv_final"], part_front=r["part_front_ref"],
                                faisables=float(np.mean(r["part_faisables"])),
                                minutes=float(np.mean(r["secondes"]) / 60))
    leg = ", ".join("{" + NOMS[m] + "}" for m in meth)
    write("echelle_hv", rf"""\begin{{tikzpicture}}
\begin{{axis}}[width=0.95\linewidth,height=6.2cm,xlabel={{profils simulés}},ylabel={{hypervolume / référence (médiane)}},
  xmin=20,xmax=300,tick label style={{font=\scriptsize}},label style={{font=\small}},
  legend style={{font=\scriptsize,draw=none,at={{(0.5,-0.22)}},anchor=north,legend columns=2}},legend cell align=left]
{chr(10).join(lines)}
\legend{{{leg}}}
\end{{axis}}
\end{{tikzpicture}}""")
    F = np.array(D["front_ref_F"])
    o = np.argsort(F[:, 0])
    write("echelle_front", rf"""\begin{{tikzpicture}}
\begin{{axis}}[width=0.7\linewidth,height=5.4cm,xlabel={{$C_D$ (counts)}},ylabel={{SER S2 (dBm, optique physique)}},
  tick label style={{font=\scriptsize}},label style={{font=\small}}]
\addplot[cblue,very thick,mark=*,mark size=1.5pt] coordinates {{{co(F[o, 0], F[o, 1])}}};
\end{{axis}}
\end{{tikzpicture}}""")


if __name__ == "__main__":
    os.makedirs(FIG, exist_ok=True)
    figures_precision()
    figures_v2()
    figures_echelle()
    with open(os.path.join(FIG, "chiffres.json"), "w") as f:
        json.dump(CH, f, indent=1, default=lambda o: o.item() if hasattr(o, "item") else str(o))
    print(json.dumps({k: v for k, v in CH.items() if k in ("v2", "echelle")}, indent=1)[:4000])
