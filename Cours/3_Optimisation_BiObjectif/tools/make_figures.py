"""Figures pgfplots du cours d'optimisation, calculées par le code du projet (Code/optim).

Usage (depuis LaTeX/) : python3 tools/make_figures.py
Lit Code/resultats/base_complete_*.csv et Code/resultats/pipeline.json
(produit par `python3 -m optim.experiences_pipeline` dans Code/).
Les chiffres cités sont écrits dans figures/chiffres.json.
"""
import json
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
PROJ = os.path.abspath(os.path.join(HERE, "..", ".."))   # racine du projet HackathonQuantique
CODE = os.path.join(PROJ, "Code")
RES = os.path.join(CODE, "resultats")
sys.path.insert(0, CODE)

from optim.problem import scenario, build_table, CountedProblem  # noqa: E402
from optim.pareto import pareto_front, hypervolume_2d, Normalizer  # noqa: E402
from optim.nsga2 import nsga2, sbx, poly_mutation  # noqa: E402
from optim.qubo_loop import build_qubo, fit_quadratic, fit_linear, predict  # noqa: E402

FIG = os.path.join(HERE, "figures")
os.makedirs(FIG, exist_ok=True)
NUM = {}
NAMES = ("S1_note", "S2_sol")
TITLES = {"S1_note": "S1 : approche de face", "S2_sol": "S2 : radar en contrebas"}


def write(name, body):
    with open(os.path.join(FIG, name + ".tex"), "w") as f:
        f.write(f"% figure générée par tools/make_figures.py : {name}\n{body}\n")


def co(xs, ys, fmt="{:.5g}"):
    return " ".join(f"({fmt.format(float(x))},{fmt.format(float(y))})" for x, y in zip(xs, ys)
                    if np.isfinite(x) and np.isfinite(y))


def table(name):
    return build_table(scenario(name), path=os.path.join(RES, f"base_complete_{name}.csv"))


def supported(Fn):
    """Points du front (normalisé) atteignables par une somme pondérée : enveloppe convexe inférieure."""
    o = np.argsort(Fn[:, 0])
    P = Fn[o]
    hull = []
    for p in P:
        while len(hull) >= 2:
            a, b = hull[-2], hull[-1]
            if (b[0] - a[0]) * (p[1] - a[1]) - (b[1] - a[1]) * (p[0] - a[0]) <= 1e-12:
                hull.pop()
            else:
                break
        hull.append(p)
    hs = {tuple(np.round(h, 12)) for h in hull}
    return np.array([tuple(np.round(Fn[i], 12)) in hs for i in range(len(Fn))])


# ------------------------------------------------------------ espace des objectifs
def fig_objectifs():
    blocks = []
    for name in NAMES:
        T = table(name)
        feas = [r for r in T if r["feasible"]]
        inf = [r for r in T if not r["feasible"]]
        F = np.array([[r["CD_counts"], r["SER_db"]] for r in feas])
        fr = pareto_front(F)
        fr = sorted(fr, key=lambda i: F[i, 0])
        Fn = Normalizer(F)(F[fr])
        sup = supported(Fn)
        tcol = {0.08: "cblue", 0.1: "cgreen", 0.12: "corange"}
        pts_by_t = {}
        for r in feas:
            pts_by_t.setdefault(round(r["t"], 2), []).append((r["CD_counts"], r["SER_db"]))
        pl = []
        for tv in sorted(pts_by_t):
            col = "cmuted!55"
            if tv == 0.08:
                col = "cblue!55"
            xy = np.array(pts_by_t[tv])
            pl.append(f"\\addplot[only marks,mark=*,mark size=0.8pt,{col}] coordinates {{{co(xy[:, 0], xy[:, 1])}}};")
        xi = np.array([[r["CD_counts"], r["SER_db"]] for r in inf])
        pl.append(f"\\addplot[only marks,mark=x,mark size=1.2pt,corange!60] coordinates {{{co(xi[:, 0], xi[:, 1])}}};")
        pl.append(f"\\addplot[cink,thick,mark=*,mark size=1.6pt,mark options={{fill=white}}] coordinates {{{co(F[fr, 0], F[fr, 1])}}};")
        NUM[name] = dict(n_feas=len(feas), n_front=len(fr), n_supported=int(sup.sum()),
                         front=[dict(m=feas[i]["m"], p=feas[i]["p"], t=feas[i]["t"], CD=F[i, 0], SER=F[i, 1]) for i in fr],
                         cd_range=[float(F[fr, 0].min()), float(F[fr, 0].max())],
                         ser_range=[float(F[fr, 1].min()), float(F[fr, 1].max())])
        blocks.append(rf"""\nextgroupplot[title={{{TITLES[name]}}}]
{chr(10).join(pl)}""")
    write("objectifs", rf"""\begin{{tikzpicture}}
\begin{{groupplot}}[group style={{group size=2 by 1,horizontal sep=1.6cm}},width=0.5\linewidth,height=6cm,
  xlabel={{$C_D$ (counts)}},ylabel={{SER (dBm)}},title style={{font=\small}},tick label style={{font=\scriptsize}}]
{chr(10).join(blocks)}
\end{{groupplot}}
\end{{tikzpicture}}""")
    # zoom sur les fronts
    blocks = []
    for name in NAMES:
        T = table(name)
        feas = [r for r in T if r["feasible"]]
        F = np.array([[r["CD_counts"], r["SER_db"]] for r in feas])
        fr = sorted(pareto_front(F), key=lambda i: F[i, 0])
        Fn = Normalizer(F)(F[fr])
        sup = supported(Fn)
        near = [i for i in range(len(F)) if feas[i]["t"] < 0.09 and F[i, 0] < F[fr, 0].max() + 6]
        lab = []
        for k, i in enumerate(fr):
            r = feas[i]
            if k % max(1, len(fr) // 6) == 0 or k == len(fr) - 1:
                lab.append(f"\\node[font=\\tiny,anchor=south west,inner sep=1pt] at (axis cs:{F[i,0]:.3f},{F[i,1]:.4f}) {{({100*r['m']:.0f}; {r['p']:.2f})}};")
        blocks.append(rf"""\nextgroupplot[title={{{TITLES[name]} : front ($t=8$~\%)}}]
\addplot[only marks,mark=*,mark size=1pt,cmuted!60] coordinates {{{co(F[near, 0], F[near, 1])}}};
\addplot[cink,thick,mark=*,mark size=1.8pt,mark options={{fill=white}}] coordinates {{{co(F[fr, 0], F[fr, 1])}}};
\addplot[only marks,mark=*,mark size=1.8pt,cblue] coordinates {{{co(F[np.array(fr)[sup], 0], F[np.array(fr)[sup], 1])}}};
{chr(10).join(lab)}""")
    write("fronts", rf"""\begin{{tikzpicture}}
\begin{{groupplot}}[group style={{group size=2 by 1,horizontal sep=1.6cm}},width=0.5\linewidth,height=6cm,
  xlabel={{$C_D$ (counts)}},ylabel={{SER (dBm)}},title style={{font=\small}},tick label style={{font=\scriptsize}},
  yticklabel style={{/pgf/number format/fixed,/pgf/number format/precision=2}}]
{chr(10).join(blocks)}
\end{{groupplot}}
\end{{tikzpicture}}""")


# ------------------------------------------------------------ opérateurs génétiques
def fig_operateurs():
    rng = np.random.default_rng(0)
    p1, p2 = np.array([2.0]), np.array([5.0])
    c = []
    for _ in range(40000):
        a, b = sbx(p1, p2, -0.5, 7.5, 15.0, rng)
        c += [a[0], b[0]]
    c = np.array(c)
    c = c[np.abs(c - 2) + np.abs(c - 5) > 1e-9]       # retire les copies sans croisement
    h, e = np.histogram(c, bins=np.linspace(-0.5, 7.5, 81), density=True)
    m = [poly_mutation(np.array([3.0]), -0.5, 7.5, 20.0, 1.0, rng)[0] for _ in range(40000)]
    hm, em = np.histogram(m, bins=np.linspace(-0.5, 7.5, 81), density=True)
    ctr, cm_ = 0.5 * (e[1:] + e[:-1]), 0.5 * (em[1:] + em[:-1])
    # probabilité de chaque indice entier après arrondi
    pi = [np.mean(np.rint(np.clip(c, -0.5, 7.49)) == k) for k in range(8)]
    pm = [np.mean(np.rint(np.clip(m, -0.5, 7.49)) == k) for k in range(8)]
    NUM["sbx_int"] = pi
    NUM["pm_int"] = pm
    write("operateurs", rf"""\begin{{tikzpicture}}
\begin{{groupplot}}[group style={{group size=2 by 1,horizontal sep=1.6cm}},width=0.5\linewidth,height=5.2cm,
  title style={{font=\small}},tick label style={{font=\scriptsize}},xmin=-0.5,xmax=7.5,xtick={{0,...,7}},ymin=0]
\nextgroupplot[title={{SBX ($\eta_c=15$), parents 2 et 5}},xlabel={{valeur de l'enfant (indice relâché)}},ylabel={{densité}}]
\addplot[cblue,very thick] coordinates {{{co(ctr, h)}}};
\addplot[only marks,mark=triangle*,mark size=3pt,corange] coordinates {{(2,0) (5,0)}};
\nextgroupplot[title={{mutation polynomiale ($\eta_m=20$), parent 3}},xlabel={{valeur après mutation}}]
\addplot[cgreen,very thick] coordinates {{{co(cm_, hm)}}};
\addplot[only marks,mark=triangle*,mark size=3pt,corange] coordinates {{(3,0)}};
\end{{groupplot}}
\end{{tikzpicture}}""")


# ------------------------------------------------------------ taille de population
def fig_population():
    out = {}
    for name in NAMES:
        T = table(name)
        feas = [r for r in T if r["feasible"]]
        F = np.array([[r["CD_counts"], r["SER_db"]] for r in feas])
        exact = {(feas[i]["im"], feas[i]["ip"], feas[i]["it"]) for i in pareto_front(F)}
        res = {}
        for pop in (8, 12, 16, 24, 32, 48):
            v = []
            for s in range(20):
                pb = CountedProblem(T)
                hit = []
                nsga2(pb, pop_size=pop, n_gen=400 // pop + 40, seed=s,
                      callback=lambda g, p: hit.append(p.n_sims) if exact <= p._set and not hit else None)
                v.append(hit[0] if hit else np.nan)
            res[pop] = [float(np.nanpercentile(v, q)) for q in (25, 50, 75)] + [float(np.mean(np.isfinite(v)))]
        out[name] = res
    NUM["population"] = out
    pl = []
    for name, st in zip(NAMES, ("cblue", "corange")):
        pops = sorted(out[name])
        med = [out[name][p][1] for p in pops]
        lo = [out[name][p][0] for p in pops]
        hi = [out[name][p][2] for p in pops]
        pl.append(f"\\addplot[{st},very thick,mark=*,mark size=1.5pt,error bars/.cd,y dir=both,y explicit] coordinates {{"
                  + " ".join(f"({p},{m_:.1f}) +- (0,0)" for p, m_ in zip(pops, med)) + "};" + f"\\addlegendentry{{{TITLES[name]}}}")
        pl.append(f"\\addplot[{st},dashed,forget plot] coordinates {{{co(pops, lo)}}};")
        pl.append(f"\\addplot[{st},dashed,forget plot] coordinates {{{co(pops, hi)}}};")
    write("population", rf"""\begin{{tikzpicture}}
\begin{{axis}}[width=0.7\linewidth,height=5.6cm,xlabel={{taille de population $N$}},ylabel={{simulations pour le front complet}},
  xmode=log,xtick={{8,12,16,24,32,48}},xticklabels={{8,12,16,24,32,48}},legend style={{draw=none,font=\scriptsize}},legend pos=north west,ymin=0]
{chr(10).join(pl)}
\end{{axis}}
\end{{tikzpicture}}""")


# ------------------------------------------------------------ résultats de la chaîne
def curve(traces, key, grid):
    """Médiane et quartiles d'un indicateur (frac ou hv) en fonction du nombre de simulations."""
    kidx = 1 if key == "frac" else 2
    M = []
    for tr in traces:
        tr = np.array(tr)
        v = []
        for g in grid:
            k = np.searchsorted(tr[:, 0], g, side="right") - 1
            v.append(tr[k, kidx] if k >= 0 else 0.0)
            if k == len(tr) - 1 and tr[-1, kidx] >= 1.0 - 1e-12:
                v[-1] = 1.0
        M.append(v)
    M = np.array(M)
    return np.percentile(M, 25, axis=0), np.median(M, axis=0), np.percentile(M, 75, axis=0)


def fig_pipeline():
    path = os.path.join(RES, "pipeline.json")
    if not os.path.exists(path):
        print("pipeline.json absent : lancer python3 -m optim.experiences_pipeline")
        return
    D = json.load(open(path))
    meth = [("random", "cmuted", "tirage aléatoire"), ("nsga2", "corange", "NSGA-II"),
            ("loop_exact", "cblue", "boucle QUBO, solveur exact"), ("loop_sa", "cgreen", "boucle QUBO, recuit simulé"),
            ("loop_qaoa", "cviolet", "boucle QUBO, QAOA $p=2$")]
    for key, fname, ylab in (("frac", "convergence_front", "fraction du front exact trouvée"),
                             ("hv", "convergence_hv", "hypervolume / hypervolume exact")):
        blocks = []
        for name in NAMES:
            grid = np.arange(0, 301, 2)
            pl = []
            for mk, col, lab in meth:
                if mk not in D[name]:
                    continue
                lo, md, hi = curve(D[name][mk], key, grid)
                pl.append(f"\\addplot[{col},very thick,forget plot] coordinates {{{co(grid, md)}}};")
                pl.append(f"\\addplot[{col},thin,dashed,forget plot] coordinates {{{co(grid, lo)}}};")
            if name == NAMES[0]:
                pl = [f"\\addlegendimage{{{col},very thick}}\\addlegendentry{{{lab}}}" for mk, col, lab in meth if mk in D[name]] + pl
            blocks.append(rf"""\nextgroupplot[title={{{TITLES[name]} ({D[name]['n_front']} profils sur le front)}}]
{chr(10).join(pl)}""")
        write(fname, rf"""\begin{{tikzpicture}}
\begin{{groupplot}}[group style={{group size=2 by 1,horizontal sep=1.4cm}},width=0.5\linewidth,height=6cm,
  xlabel={{simulations distinctes}},ylabel={{{ylab}}},xmin=0,xmax=300,ymin=0,ymax=1.02,
  title style={{font=\small}},tick label style={{font=\scriptsize}},
  legend style={{draw=none,font=\scriptsize,at={{(1.1,-0.25)}},anchor=north}},legend columns=3,legend cell align=left]
{chr(10).join(blocks)}
\end{{groupplot}}
\end{{tikzpicture}}""")
    NUM["summary"] = {n: D[n]["summary"] for n in NAMES}


def fig_qubo_example():
    """Un QUBO de la boucle : nombre de qubits, de couplages, qualité des modèles après 30 simulations."""
    T = table("S2_sol")
    rng = np.random.default_rng(3)
    keys = [(r["im"], r["ip"], r["it"]) for r in T]
    pick = [T[j] for j in rng.permutation(len(T))[:30]]
    ok = [r for r in pick if r["feasible"]]
    I = np.array([[r["im"], r["ip"], r["it"]] for r in ok])
    cd = np.array([r["CD_counts"] for r in ok])
    ser = np.array([r["SER_db"] for r in ok])
    val = [r for r in pick if r["valid"]]
    Iv = np.array([[r["im"], r["ip"], r["it"]] for r in val])
    c_cd, c_ser = fit_quadratic(I, cd), fit_quadratic(I, ser)
    c_cl = fit_linear(Iv, np.array([r["CL"] for r in val]))
    Q, c, n = build_qubo(c_cd, c_ser, c_cl, 0.5, np.ptp(cd), np.ptp(ser), {"t_min_index": True, "cl_min": 0.3})
    nz = int(np.count_nonzero(np.abs(np.triu(Q, 1)) > 1e-12))
    allI = np.array([[r["im"], r["ip"], r["it"]] for r in T if r["feasible"]])
    allcd = np.array([r["CD_counts"] for r in T if r["feasible"]])
    allser = np.array([r["SER_db"] for r in T if r["feasible"]])
    r2 = lambda y, yh: 1 - np.sum((y - yh) ** 2) / np.sum((y - y.mean()) ** 2)
    NUM["qubo_example"] = dict(n_qubits=n, n_couplings=nz, n_cnot=2 * nz,
                               r2_cd=float(r2(allcd, predict(c_cd, allI))), r2_ser=float(r2(allser, predict(c_ser, allI))),
                               n_train=len(ok))
    # matrice Q en carte
    M = np.triu(Q) + np.triu(Q, 1).T
    vmax = np.abs(M).max()
    pts = " ".join(f"({j},{i}) [{M[i, j] / vmax:.4f}]" for i in range(n) for j in range(n))
    labels = ["$m_0$", "$m_1$", "$m_2$", "$p_0$", "$p_1$", "$p_2$", "$t_0$", "$t_1$", "$t_2$", "$y$", "$s_0$", "$s_1$", "$s_2$"][:n]
    write("matrice_Q", rf"""\begin{{tikzpicture}}
\begin{{axis}}[width=7.4cm,height=7.4cm,enlargelimits=false,axis on top,y dir=reverse,
  xtick={{0,...,{n-1}}},ytick={{0,...,{n-1}}},xticklabels={{{",".join(labels)}}},yticklabels={{{",".join(labels)}}},
  tick label style={{font=\scriptsize}},colorbar,colorbar style={{width=2.5mm,tick label style={{font=\scriptsize}}}},
  colormap={{div}}{{color=(corange) color=(white) color=(cblue)}},point meta min=-1,point meta max=1]
\addplot[matrix plot*,mesh/cols={n},point meta=explicit] coordinates {{{pts}}};
\end{{axis}}
\end{{tikzpicture}}""")


if __name__ == "__main__":
    only = sys.argv[1:]
    for f in (fig_objectifs, fig_operateurs, fig_population, fig_qubo_example, fig_pipeline):
        if only and f.__name__ not in only:
            continue
        f()
        print("ok", f.__name__)
    path = os.path.join(FIG, "chiffres.json")
    old = json.load(open(path)) if os.path.exists(path) else {}
    old.update(NUM)
    with open(path, "w") as fh:
        json.dump(old, fh, indent=1, default=float)
    print(json.dumps({k: v for k, v in NUM.items() if k not in ("S1_note", "S2_sol")}, indent=1, default=float)[:3000])
