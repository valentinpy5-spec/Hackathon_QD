"""Itération V3, lot L6 : précision des modèles en fonction du nombre de profils simulés.

Pour chaque grandeur (C_D, SER, C_L,max), chaque modèle et chaque taille N de jeu d'entraînement, on tire
10 jeux au hasard, on ajuste, et l'on mesure sur les profils non utilisés :
    - l'erreur quadratique moyenne, divisée par l'écart-type de la grandeur (0 = parfait, 1 = aussi bon
      que de prédire la moyenne) ;
    - la corrélation de rang de Spearman (le classement compte plus que la valeur pour optimiser).
Deux jeux de données : la grille V2 (1 024 profils, toute la table) et le problème CST (profils du cache).

    python3 -m ia_qubo.precision_modeles          -> resultats/ia_qubo_precision.json
"""
import json
import os
import sys

import numpy as np
from scipy.stats import spearmanr

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from ia_qubo.modeles import QuadZ, FactorizationMachine, BayesQuadBits, MLPEnsemble, encode  # noqa: E402

HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUT = os.path.join(HERE, "resultats", "ia_qubo_precision.json")
MODELES = ("quadz", "fm", "bocs", "mlp")


def donnees_v2():
    from optim.problem_v2 import build_table_v2, with_objective, SpaceV2, KEYS
    SP = SpaceV2()
    T = with_objective(build_table_v2(), "S2", "MoM")
    return T, KEYS, SP.bits, SP.n_levels


def donnees_cst():
    from ia_qubo import probleme_cst as P
    return list(P._charge_cache().values()), P.KEYS, P.SPEC["bits"], P.SPEC["levels"]


def fabrique(nom, bits, levels, seed):
    n = sum(bits)
    return {"quadz": lambda: QuadZ(bits, levels),
            "fm": lambda: FactorizationMachine(n, k=4, seed=seed),
            "bocs": lambda: BayesQuadBits(n, seed=seed),
            "mlp": lambda: MLPEnsemble(bits, levels, epochs=1000, n_models=4, seed=seed)}[nom]()


def etude(rows, keys, bits, levels, tailles, n_rep=10):
    out = {}
    cibles = {"CD_counts": lambda r: r["feasible"] and r["CD_counts"] < 200,
              "SER_db": lambda r: r["feasible"],
              "CLmax": lambda r: r["valid"] and np.isfinite(r["CLmax"])}
    for cible, filtre in cibles.items():
        R = [r for r in rows if filtre(r)]
        B = encode(np.array([[r[k] for k in keys] for r in R]), bits)
        y = np.array([r[cible] for r in R], float)
        sd = y.std()
        out[cible] = {"n_donnees": len(R)}
        for nom in MODELES:
            err, rho = [], []
            for N in tailles:
                if N >= len(R) - 20:
                    err.append(None)
                    rho.append(None)
                    continue
                e_, r_ = [], []
                for rep in range(n_rep):
                    rng = np.random.default_rng(1000 * rep + N)
                    p = rng.permutation(len(R))
                    tr, te = p[:N], p[N:]
                    m = fabrique(nom, bits, levels, rep).fit(B[tr], y[tr])
                    pr = m.predict(B[te])
                    e_.append(float(np.sqrt(np.mean((pr - y[te]) ** 2)) / sd))
                    r_.append(float(spearmanr(pr, y[te]).correlation))
                err.append(float(np.median(e_)))
                rho.append(float(np.median(r_)))
            out[cible][nom] = dict(err=err, rho=rho)
            print(f"  {cible:9s} {nom:5s} erreur relative {[None if e is None else round(e, 2) for e in err]}, "
                  f"rang {[None if r is None else round(r, 2) for r in rho]}", flush=True)
    return out


if __name__ == "__main__":
    tailles = [10, 15, 20, 30, 50, 80, 120, 200, 300, 500]
    res = dict(tailles=tailles)
    quoi = [q for q in ("v2", "cst") if q in sys.argv] or ["v2", "cst"]
    for q in quoi:
        print(q, flush=True)
        res[q] = etude(*(donnees_v2() if q == "v2" else donnees_cst()), tailles)
    if os.path.exists(OUT):
        old = json.load(open(OUT))
        old.update(res)
        res = old
    with open(OUT, "w") as f:
        json.dump(res, f)
    print("->", OUT)
