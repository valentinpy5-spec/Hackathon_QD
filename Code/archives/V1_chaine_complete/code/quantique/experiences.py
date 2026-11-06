"""Expériences numériques de référence pour le cours de calcul quantique.

Toutes les valeurs citées dans le cours (et les données de ses figures TikZ)
sortent de ce script : resultats/experiences_quantique.json.

    python3 -m quantique.experiences        (depuis Code/)

A. QUBO jouet à 3 qubits (« pas deux voisins actifs ») : énergies, Ising,
   spectre adiabatique, paysage QAOA p = 1.
B. QUBO aéro à 9 qubits (modèle quadratique de CD sur les 512 profils) :
   QAOA p = 1..5, recuit simulé.
C. Boucle itérative « modèle quadratique -> QUBO -> simulation des candidats »
   sur CD seul, comparée à une recherche aléatoire.
"""
import csv
import json
import os

import numpy as np

from .qubo import (quadratic_features, surrogate_to_qubo, qubo_energy, brute_force,
                   qubo_to_ising, all_bitstrings)
from .qaoa import qaoa_state, optimize_qaoa, interp_init, normalize, expectation
from .recuit import simulated_annealing

HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUT = os.path.join(HERE, "resultats", "experiences_quantique.json")

X = np.array([[0, 1], [1, 0]], dtype=complex)


def sum_x(n):
    """Matrice 2^n x 2^n de sum_k X_k (bit k = (x >> k) & 1)."""
    N = 2**n
    M = np.zeros((N, N))
    for x in range(N):
        for k in range(n):
            M[x ^ (1 << k), x] += 1
    return M


# --------------------------------------------------------------------- A
def toy():
    Q = np.array([[-1.0, 2, 0], [0, -1, 2], [0, 0, -1]])
    c = 0.0
    n = 3
    B, E = brute_force(Q, c)
    h, J, off = qubo_to_ising(Q, c)
    # spectre de H(s) = -(1-s) sum X + s H_C
    HX = sum_x(n)
    s_grid = np.linspace(0, 1, 101)
    spec = np.array([np.linalg.eigvalsh(-(1 - s) * HX + s * np.diag(E)) for s in s_grid])
    gap = spec[:, 1] - spec[:, 0]
    # paysage p = 1 (énergie brute, déjà d'ordre 1)
    # énergies entières : période 2 pi en gamma ; beta -> beta + pi ne change
    # qu'une phase globale, donc période pi en beta
    gs = np.linspace(0, 2 * np.pi, 73)
    bs = np.linspace(0, np.pi, 37)
    land = np.array([[expectation(np.array([g, b]), E, n) for g in gs] for b in bs])
    r = optimize_qaoa(E, n, 1, n_restarts=20)
    psi = qaoa_state(r.x[:1], r.x[1:], E, n)
    k_opt = int(np.argmin(E))
    res = dict(Q=Q.tolist(), E=E.tolist(), bits=B.astype(int).tolist(),
               h=h.tolist(), J=J.tolist(), offset=off,
               s=s_grid.tolist(), spectrum=spec.tolist(),
               gap_min=float(gap.min()), s_gap_min=float(s_grid[np.argmin(gap)]),
               land_gamma=gs.tolist(), land_beta=bs.tolist(), landscape=land.tolist(),
               p1_params=r.x.tolist(), p1_energy=r.fun,
               p1_prob=(np.abs(psi) ** 2).tolist(), opt_index=k_opt)
    # p = 2 et 3 pour montrer la convergence vers l'état fondamental
    prev = r.x
    for p in (2, 3):
        rp = optimize_qaoa(E, n, p, n_restarts=5, init=interp_init(prev, p))
        prev = rp.x
        ps = qaoa_state(rp.x[:p], rp.x[p:], E, n)
        res[f"p{p}_prob_opt"] = float(np.abs(ps[k_opt]) ** 2)
        res[f"p{p}_energy"] = rp.fun
    return res


# --------------------------------------------------------------------- B
def load_base():
    path = os.path.join(HERE, "resultats", "base_aero_Re3e+06_a2.csv")
    R = list(csv.DictReader(open(path)))
    I = np.array([[int(r["im"]), int(r["ip"]), int(r["it"])] for r in R])
    CD = np.array([float(r["CD"]) for r in R]) * 1e4      # counts
    bits = np.array([[int(ch) for ch in r["bits"]] for r in R])
    # index de vecteur d'état x = sum_k b_k 2^k
    xidx = bits @ (1 << np.arange(9))
    CD_by_x = np.empty(512)
    CD_by_x[xidx] = CD
    return I, CD, CD_by_x


def aero():
    I, CD, CD_by_x = load_base()
    F = quadratic_features(I / 7 * 2 - 1)
    coef, *_ = np.linalg.lstsq(F, CD, rcond=None)
    Q, c = surrogate_to_qubo(coef, [3, 3, 3], [8, 8, 8])
    B, E = brute_force(Q, c)
    n = 9
    k = int(np.argmin(E))
    true_rank = int(np.argsort(np.argsort(CD_by_x))[k])     # 0 = meilleur réel
    En = normalize(E)
    order = np.argsort(E)
    out = dict(n=n, qubo_min=float(E.min()), qubo_second=float(E[order[1]]),
               qubo_second_bits=B[order[1]].astype(int).tolist(), qubo_argmin_bits=B[k].astype(int).tolist(),
               qubo_argmin_true_rank=true_rank, true_min=float(CD_by_x.min()),
               true_at_qubo_min=float(CD_by_x[k]),
               n_nonzero_offdiag=int(np.count_nonzero(np.triu(Q, 1))),
               qaoa=[])
    prev = None
    for p in range(1, 6):
        init = interp_init(prev, p) if prev is not None else None
        r = optimize_qaoa(En, n, p, n_restarts=8 if p == 1 else 3, init=init)
        prev = r.x
        psi = qaoa_state(r.x[:p], r.x[p:], En, n)
        pr = np.abs(psi) ** 2
        Em = (E - E.min()) / (E.max() - E.min())
        out["qaoa"].append(dict(p=p, energy=r.fun, prob_opt=float(pr[k]),
                                prob_top5=float(pr[order[:5]].sum()),
                                ratio=float(1 - pr @ Em),
                                gammas=r.x[:p].tolist(), betas=r.x[p:].tolist(),
                                prob_by_rank=pr[order[:25]].tolist(),
                                shots_to_opt=float(1 / pr[k])))
    # sans normalisation : le piège de l'échelle
    r_raw = optimize_qaoa(E, n, 1, n_restarts=8)
    psi = qaoa_state(r_raw.x[:1], r_raw.x[1:], E, n)
    out["raw_p1_prob_opt"] = float(np.abs(psi[k]) ** 2)
    # recuit simulé
    sa = []
    for sweeps in (2, 5, 10, 20, 50, 100):
        _, _, fin = simulated_annealing(Q, c, n_sweeps=sweeps, n_reads=200, seed=1)
        sa.append(dict(sweeps=sweeps, success=float(np.mean(np.isclose(fin, E.min()))),
                       success_01=float(np.mean(fin - E.min() < 0.1))))
    out["sa"] = sa
    return out


# --------------------------------------------------------------------- C
def boucle(n_seeds=50, n_init=10, batch=3, budget=200):
    """Recherche du vrai minimum de CD par modèle quadratique itératif.

    Le solveur de QUBO est ici l'énumération exacte du QUBO : en phase 4 il
    sera remplacé par QAOA ou le recuit, qui doivent retrouver les mêmes
    candidats de basse énergie.
    """
    I, CD, _ = load_base()
    Z = I / 7 * 2 - 1
    F = quadratic_features(Z)
    best_true = int(np.argmin(CD))
    found, curves = [], []
    for seed in range(n_seeds):
        rng = np.random.default_rng(seed)
        D = list(rng.choice(len(CD), n_init, replace=False))
        curve = [CD[D].min()]
        while best_true not in D and len(D) < budget:
            coef, *_ = np.linalg.lstsq(F[D], CD[D], rcond=None)
            pred = F @ coef                      # = énergie du QUBO pour chaque profil
            cand = [i for i in np.argsort(pred) if i not in D][:batch]
            D += cand
            curve.append(CD[D].min())
        found.append(len(D))
        curves.append([float(CD[D[:m]].min()) for m in range(n_init, min(len(D), 60) + 1)])
    L = max(len(cv) for cv in curves)
    padded = np.array([cv + [cv[-1]] * (L - len(cv)) for cv in curves])
    # recherche aléatoire : médiane de la meilleure valeur après m tirages
    rnd = []
    for seed in range(n_seeds):
        perm = np.random.default_rng(1000 + seed).permutation(len(CD))
        rnd.append(np.minimum.accumulate(CD[perm])[n_init - 1:60])
    return dict(n_init=n_init, batch=batch, sims_to_opt=found,
                median_sims=float(np.median(found)), q25=float(np.percentile(found, 25)),
                q75=float(np.percentile(found, 75)), random_expected=(len(CD) + 1) / 2,
                true_min=float(CD.min()),
                median_curve=np.median(padded, axis=0).tolist(),
                random_curve=np.median(np.array(rnd), axis=0).tolist())


if __name__ == "__main__":
    res = dict(toy=toy(), aero=aero(), boucle=boucle())
    with open(OUT, "w") as f:
        json.dump(res, f, indent=1)
    t, a, b = res["toy"], res["aero"], res["boucle"]
    print("Jouet : E =", t["E"], " gap min %.3f en s = %.2f" % (t["gap_min"], t["s_gap_min"]))
    print("        p=1 P(opt) = %.3f, p=2 %.3f, p=3 %.3f" % (t["p1_prob"][t["opt_index"]], t["p2_prob_opt"], t["p3_prob_opt"]))
    print("Aéro : min QUBO %.2f counts, vrai CD à ce point %.2f (rang réel %d), vrai min %.2f"
          % (a["qubo_min"], a["true_at_qubo_min"], a["qubo_argmin_true_rank"], a["true_min"]))
    for q in a["qaoa"]:
        print("  QAOA p=%d  P(opt)=%.3f  P(top5)=%.3f  r=%.3f  1/P=%.0f" % (q["p"], q["prob_opt"], q["prob_top5"], q["ratio"], q["shots_to_opt"]))
    print("  sans normalisation p=1 : P(opt) = %.4f" % a["raw_p1_prob_opt"])
    print("  recuit (balayages, succès exact, à 0.1 count) :", [(s["sweeps"], s["success"], s["success_01"]) for s in a["sa"]])
    print("  deux plus bas niveaux du QUBO :", sorted(a["qaoa"][0]["prob_by_rank"])[-2:])
    print("Boucle : médiane %.0f simulations (q25 %.0f, q75 %.0f) contre %.0f en aléatoire"
          % (b["median_sims"], b["q25"], b["q75"], b["random_expected"]))
