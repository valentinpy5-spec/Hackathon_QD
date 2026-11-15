"""Figures et chiffres du cours d'optimisation bayésienne (itération V3, lot L1).

Usage (depuis le dossier du cours, avec l'environnement Python du projet) :
    python tools/make_figures.py
Lit Code/resultats/pipeline_L1_V3.json (python -m optim.experiences_l1) et
Code/resultats/diagnostic_L1_V3.json (python -m optim.diagnostic_l1).
Écrit figures/*.tex et figures/chiffres.json (les nombres cités dans le texte).
"""
import json
import os
import sys
import warnings

import numpy as np
from scipy.stats import wilcoxon

HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
PROJ = os.path.abspath(os.path.join(HERE, "..", ".."))
RES = os.path.join(PROJ, "Code", "resultats")
sys.path.insert(0, os.path.join(PROJ, "Code"))
warnings.filterwarnings("ignore")

from optim.bayes import GPModel, QuadModel, expected_improvement  # noqa: E402

FIG = os.path.join(HERE, "figures")
TITLES = {"S1": "S1 : approche de face", "S2": "S2 : radar en contrebas"}
BUDGET = 250
NUM = {}

# méthode -> (couleur, style de trait, légende)
METH = [
    ("loop_exact", "cink", "solid", "boucle QUBO (V2)"),
    ("quad_mean", "cblue", "solid", "quadratique + minimum"),
    ("gp_mean", "cblue", "densely dashed", "krigeage + minimum"),
    ("quad_ei", "corange", "solid", "quadratique + EI"),
    ("gp_ei", "corange", "densely dashed", "krigeage + EI"),
    ("parego", "cgreen", "densely dashed", "ParEGO"),
]
STRICT = [("quad_mean_pof", "cmuted", "solid", "quadratique + minimum, contrainte stricte"),
          ("gp_ei_dur", "cmuted", "densely dashed", "krigeage + EI, contrainte en dur")]


def write(name, body):
    with open(os.path.join(FIG, name + ".tex"), "w", encoding="utf-8") as f:
        f.write(f"% figure générée par tools/make_figures.py : {name}\n{body}\n")


def co(xs, ys):
    return " ".join(f"({float(x):.5g},{float(y):.5g})" for x, y in zip(xs, ys) if np.isfinite(x) and np.isfinite(y))


def first(tr, k, thr):
    for row in tr:
        if row[k] >= thr - 1e-12:
            return row[0]
    return np.inf


# ------------------------------------------------------------ figure 1 : krigeage et EI en 1D
def fig_gp_1d():
    def f(x):
        return 0.6 * np.sin(6.0 * x) + x**2
    X = np.array([-1.0, 0.1, 0.3, 0.5, 0.7, 0.9])       # un grand intervalle sans simulation à gauche
    y = f(X)
    g = GPModel(X.reshape(-1, 1), y)
    xs = np.linspace(-1, 1, 161)
    mu, sd = g.predict(xs.reshape(-1, 1))
    ei = expected_improvement(mu, sd, y.min())
    band = co(np.concatenate([xs, xs[::-1]]), np.concatenate([mu + 2 * sd, (mu - 2 * sd)[::-1]]))
    write("gp_1d", rf"""\begin{{tikzpicture}}
\begin{{groupplot}}[group style={{group size=1 by 2,vertical sep=0.5cm,xticklabels at=edge bottom}},width=0.82\linewidth,
  xmin=-1,xmax=1,tick label style={{font=\scriptsize}},label style={{font=\small}},
  legend style={{draw=none,font=\scriptsize,at={{(0.5,1.03)}},anchor=south}},legend columns=4]
\nextgroupplot[height=5.6cm,ylabel={{$f(x)$}}]
\addplot[fill=cblue!15,draw=none,forget plot] coordinates {{{band}}};
\addplot[cmuted,thick,dashed] coordinates {{{co(xs, f(xs))}}};\addlegendentry{{fonction}}
\addplot[cblue,very thick] coordinates {{{co(xs, mu)}}};\addlegendentry{{prédiction $\mu$}}
\addplot[only marks,mark=*,mark size=2pt,cink] coordinates {{{co(X, y)}}};\addlegendentry{{simulations}}
\addlegendimage{{area legend,fill=cblue!15,draw=none}}\addlegendentry{{$\mu\pm2s$}}
\nextgroupplot[height=3.6cm,xlabel={{$x$}},ylabel={{EI}},ymin=0]
\addplot[corange,very thick,fill=corange!15] coordinates {{{co(xs, ei)}}} \closedcycle;
\end{{groupplot}}
\end{{tikzpicture}}""")
    pk = [float(xs[i]) for i in range(1, len(xs) - 1) if ei[i] > ei[i - 1] and ei[i] >= ei[i + 1] and ei[i] > 0.2 * ei.max()]
    NUM["gp_1d"] = dict(x_ei_max=float(xs[np.argmax(ei)]), x_mu_min=float(xs[np.argmin(mu)]), f_min=float(y.min()),
                        x_true_min=float(xs[np.argmin(f(xs))]), ei_peaks=pk)


# ------------------------------------------------------------ validation sur un cas jouet (chapitre code)
def toy_validation():
    rng = np.random.default_rng(0)

    def f(Z):
        return np.sin(2 * Z[:, 0]) + Z[:, 1] ** 2 - 0.5 * Z[:, 2] * Z[:, 3]
    Z, Zt = rng.uniform(-1, 1, (60, 4)), rng.uniform(-1, 1, (500, 4))
    y = f(Z) + 0.01 * rng.normal(size=60)
    out = {}
    for name, M in (("gp", GPModel), ("quad", QuadModel)):
        mu, sd = M(Z, y).predict(Zt)
        err = mu - f(Zt)
        out[name] = dict(rmse=float(np.sqrt(np.mean(err**2))), cover=float(np.mean(np.abs(err) <= 2 * sd)))
    out["std_f"] = float(np.std(f(Zt)))
    out["ei_check"] = float(expected_improvement(np.array([0.0]), np.array([1.0]), 0.0)[0])
    NUM["toy"] = out


# ------------------------------------------------------------ figures 2 et 3 : la comparaison
def fig_comparison():
    D = json.load(open(os.path.join(RES, "pipeline_L1_V3.json")))
    grid = np.arange(0, BUDGET + 1, 2)
    blocks, bars = [], []
    NUM["comparison"] = {}
    for sec in ("S1", "S2"):
        R = D[sec]
        ff = {m: np.array([first(tr, 1, 1.0) for tr in R[m]]) for m, *_ in METH + STRICT}
        ref = np.where(np.isfinite(ff["loop_exact"]), ff["loop_exact"], BUDGET + 1)
        num = {}
        for m, *_ in METH + STRICT:
            c = np.where(np.isfinite(ff[m]), ff[m], BUDGET + 1)
            q = np.percentile(c, [25, 50, 75])
            p = float(wilcoxon(c, ref).pvalue) if m != "loop_exact" and np.any(c != ref) else None
            s = R["summary"][m]
            num[m] = dict(q1=q[0], median=q[1], q3=q[2], success=float(np.mean(np.isfinite(ff[m]))), p_vs_qubo=p,
                          stop_sims=s["stop_sims"][1], stop_frac=s["stop_frac"][1], stop_hv=s["stop_hv"][1],
                          seconds=float(np.mean(R[m + "_seconds"])))
        NUM["comparison"][sec] = dict(n_front=R["n_front"], n_seeds=R["n_seeds"], methods=num)
        # convergence : fraction du front trouvée (médiane)
        pl = []
        if sec == "S1":
            pl += [f"\\addlegendimage{{{c},{st},very thick}}\\addlegendentry{{{l}}}" for m, c, st, l in METH]
        for m, c, st, l in METH:
            M = []
            for tr in R[m]:
                tr = np.array(tr)
                M.append([tr[j, 1] if (j := np.searchsorted(tr[:, 0], g, side="right") - 1) >= 0 else 0.0 for g in grid])
            pl.append(f"\\addplot[{c},{st},very thick,forget plot] coordinates {{{co(grid, np.median(M, axis=0))}}};")
        blocks.append(rf"""\nextgroupplot[title={{{TITLES[sec]} ({R['n_front']} profils sur le front)}}]
{chr(10).join(pl)}""")
        # barres : médiane et quartiles du nombre de simulations pour le front complet
        allm = METH + STRICT
        pl = []
        for k, (m, c, st, l) in enumerate(allm):
            yk = len(allm) - k
            v = num[m]
            if v["success"] < 0.5:
                pl.append(f"\\node[font=\\scriptsize,{c},anchor=west] at (axis cs:8,{yk}) {{échec ({100 * v['success']:.0f}~\\% de réussite)}};")
                continue
            pl.append(f"\\addplot[{c},line width=2.2pt,forget plot] coordinates {{({v['q1']:.4g},{yk}) ({min(v['q3'], BUDGET):.4g},{yk})}};")
            pl.append(f"\\addplot[only marks,mark=*,mark size=2.4pt,{c},mark options={{fill=white,line width=1pt}},forget plot] coordinates {{({v['median']:.4g},{yk})}};")
        ticks = ",".join(str(len(allm) - k) for k in range(len(allm)))
        labels = ",".join("{" + l + "}" for *_, l in allm) if sec == "S1" else ""
        bars.append(rf"""\nextgroupplot[title={{{TITLES[sec]}}},ytick={{{ticks}}},yticklabels={{{labels}}}]
{chr(10).join(pl)}""")
    write("l1_convergence", rf"""\begin{{tikzpicture}}
\begin{{groupplot}}[group style={{group size=2 by 1,horizontal sep=1.4cm}},width=0.5\linewidth,height=6cm,
  xlabel={{simulations distinctes}},ylabel={{fraction du front exact trouvée}},xmin=0,xmax=250,ymin=0,ymax=1.02,
  title style={{font=\small}},tick label style={{font=\scriptsize}},label style={{font=\small}},
  legend style={{draw=none,font=\scriptsize,at={{(1.1,-0.25)}},anchor=north}},legend columns=3,legend cell align=left]
{chr(10).join(blocks)}
\end{{groupplot}}
\end{{tikzpicture}}""")
    write("l1_sims", rf"""\begin{{tikzpicture}}
\begin{{groupplot}}[group style={{group size=2 by 1,horizontal sep=0.5cm}},width=0.4\linewidth,height=6.4cm,
  xlabel={{simulations pour le front complet}},xmin=0,xmax=250,ymin=0.4,ymax={len(METH) + len(STRICT) + 0.6},
  title style={{font=\small}},tick label style={{font=\scriptsize}},label style={{font=\small}},
  yticklabel style={{font=\scriptsize,align=right}},xmajorgrids,grid style={{cmuted!25}}]
{chr(10).join(bars)}
\end{{groupplot}}
\end{{tikzpicture}}""")


# ------------------------------------------------------------ figures 4 et 5 : les diagnostics
DIAG = [  # nom, étiquette, couleur, marque
    ("qubo_x2.5", "1", "cink", "square*"), ("qubo_x25", "2", "cink", "square*"), ("qubo_x250", "3", "cink", "square*"),
    ("qubo_sans", "4", "cink", "square*"), ("quad_mean_lin", "5", "cblue", "*"), ("quad_mean_pof", "6", "cblue", "*"),
    ("quad_mean_sans", "7", "cblue", "*"), ("quad_ei_pof", "8", "corange", "triangle*"), ("quad_ei_sans", "9", "corange", "triangle*"),
]


def fig_diagnostic():
    path = os.path.join(RES, "diagnostic_L1_V3.json")
    if not os.path.exists(path):
        print("diagnostic_L1_V3.json absent : lancer python -m optim.diagnostic_l1")
        return
    D = json.load(open(path))
    blocks = []
    NUM["constraint"] = {}
    for sec in ("S1", "S2"):
        C = D["constraint"][sec]
        num = {"base_rate": C["base_rate"]}
        pl = [f"\\draw[cmuted,dashed] (axis cs:{100 * C['base_rate']:.3g},0) -- (axis cs:{100 * C['base_rate']:.3g},290);"]
        for name, lab, c, mk in DIAG:
            n = np.array([x if x is not None else np.inf for x in C[name]["n"]])
            cc = np.where(np.isfinite(n), n, BUDGET + 1)
            viol, succ, med = 100 * float(np.mean(C[name]["viol"])), float(np.mean(np.isfinite(n))), float(np.median(cc))
            num[name] = dict(viol=viol, success=succ, median=med, q1=float(np.percentile(cc, 25)), q3=float(np.percentile(cc, 75)))
            yv = med if succ >= 0.5 else 270.0
            pl.append(f"\\addplot[only marks,mark={mk},mark size=2.6pt,{c},forget plot] coordinates {{({viol:.3g},{yv:.4g})}};")
            pl.append(f"\\node[font=\\scriptsize,anchor=south west,inner sep=2pt] at (axis cs:{viol:.3g},{yv:.4g}) {{{lab}}};")
        NUM["constraint"][sec] = num
        ylab = ",ylabel={simulations pour le front complet}" if sec == "S1" else ""
        blocks.append(rf"""\nextgroupplot[title={{{TITLES[sec]}}}{ylab}]
{chr(10).join(pl)}""")
    write("diag_contrainte", rf"""\begin{{tikzpicture}}
\begin{{groupplot}}[group style={{group size=2 by 1,horizontal sep=1.1cm}},width=0.5\linewidth,height=6.2cm,
  xlabel={{profils proposés infaisables (\%)}},
  xmin=0,xmax=85,ymin=0,ymax=290,ytick={{0,50,100,150,200,250,270}},yticklabels={{0,50,100,150,200,250,échec}},
  title style={{font=\small}},tick label style={{font=\scriptsize}},label style={{font=\small}}]
{chr(10).join(blocks)}
\end{{groupplot}}
\end{{tikzpicture}}""")
    # qualité des modèles
    Q = D["quality"]
    sizes = sorted(int(k) for k in Q)
    cols = [("CD_counts", "traînée $C_D$"), ("SER_db", "SER"), ("CLmax", "$C_{L,\\max}$")]
    blocks = []
    NUM["quality"] = {}
    for k, (col, title) in enumerate(cols):
        pl = []
        for mname, c, st, leg in (("quad", "cblue", "solid", "modèle quadratique"), ("gp", "corange", "densely dashed", "krigeage")):
            med = [float(np.median(Q[str(n)][f"{col}_{mname}"]["rmse_rel"])) for n in sizes]
            cov = [float(np.median(Q[str(n)][f"{col}_{mname}"]["cover"])) for n in sizes]
            NUM["quality"][f"{col}_{mname}"] = dict(sizes=sizes, rmse_rel=med, cover=cov)
            pl.append(f"\\addplot[{c},{st},very thick,mark=*,mark size=1.6pt] coordinates {{{co(sizes, med)}}};"
                      + (f"\\addlegendentry{{{leg}}}" if k == 0 else ""))
        blocks.append(rf"""\nextgroupplot[title={{{title}}}{',ylabel={erreur relative de prédiction}' if k == 0 else ''}]
{chr(10).join(pl)}""")
    write("diag_qualite", rf"""\begin{{tikzpicture}}
\begin{{groupplot}}[group style={{group size=3 by 1,horizontal sep=1.0cm}},width=0.37\linewidth,height=5.2cm,
  xlabel={{profils d'entraînement}},xmode=log,log basis x=2,xtick={{20,40,80,160}},xticklabels={{20,40,80,160}},
  ymin=0,title style={{font=\small}},tick label style={{font=\scriptsize}},label style={{font=\small}},
  legend style={{draw=none,font=\scriptsize,at={{(1.75,-0.32)}},anchor=north}},legend columns=2]
{chr(10).join(blocks)}
\end{{groupplot}}
\end{{tikzpicture}}""")


if __name__ == "__main__":
    fig_gp_1d()
    toy_validation()
    fig_comparison()
    fig_diagnostic()
    with open(os.path.join(FIG, "chiffres.json"), "w", encoding="utf-8") as f:
        json.dump(NUM, f, indent=1, default=float)
    for sec in ("S1", "S2"):
        print(sec)
        for m, v in NUM["comparison"][sec]["methods"].items():
            p = "   -  " if v["p_vs_qubo"] is None else f"{v['p_vs_qubo']:.4f}"
            print(f"  {m:14s} {v['median']:6.1f} [{v['q1']:5.1f}-{v['q3']:5.1f}] réussite {100 * v['success']:3.0f} %  p = {p}  "
                  f"arrêt {v['stop_sims']:5.1f} sim., front {v['stop_frac']:.2f}, HV {v['stop_hv']:.3f}  {v['seconds']:5.1f} s")
    print(json.dumps({k: NUM[k] for k in ("toy", "gp_1d", "constraint", "quality") if k in NUM}, indent=1, default=float))
