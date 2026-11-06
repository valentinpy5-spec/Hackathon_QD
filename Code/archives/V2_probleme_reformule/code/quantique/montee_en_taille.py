"""V2 : comment évoluent la qualité de QAOA et les ressources quand le QUBO grandit ?

On garde le MÊME problème physique (modèle quadratique de la somme pondérée traînée/SER,
ajusté sur la base exhaustive V1, scénario S1, w = 0,5) et on affine la grille :
b bits par variable, d = 3 variables, soit n = 3 b qubits (6 à 18).

Pour chaque n :
    - énergies exactes des 2^n chaînes (référence) ;
    - QAOA p = 1, 2, 3 par vecteur d'état exact, angles optimisés (petits n) puis
      transférés de la taille précédente (grands n, comme on le ferait sur machine) ;
    - indicateurs : probabilité de l'optimum, probabilité d'une chaîne du 1 % meilleur,
      rapport d'approximation, tirs pour voir l'optimum avec 95 % de chances ;
    - coût du circuit : couplages non nuls, CNOT par couche (graphe complet) ;
    - recuit simulé : probabilité qu'une lecture trouve l'optimum.

    python3 -m quantique.montee_en_taille    -> resultats/montee_en_taille.json
"""
import json
import os
import time

import numpy as np

from .qubo import surrogate_to_qubo, all_bitstrings, qubo_energy
from . import qaoa as qa
from .recuit import simulated_annealing

HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUT = os.path.join(HERE, "resultats", "montee_en_taille.json")


def surrogate_coef():
    """Modèle quadratique (10 coef.) de 0,5 C_D/ΔC_D + 0,5 SER/ΔSER, ajusté sur les profils faisables V1 (S1)."""
    import csv
    from optim.qubo_loop import fit_quadratic
    rows = [r for r in csv.DictReader(open(os.path.join(HERE, "resultats", "base_complete_S1_note.csv")))
            if r["feasible"] == "True"]
    I = np.array([[int(r["im"]), int(r["ip"]), int(r["it"])] for r in rows])
    cd = np.array([float(r["CD_counts"]) for r in rows])
    ser = np.array([float(r["SER_db"]) for r in rows])
    return 0.5 * fit_quadratic(I, cd) / np.ptp(cd) + 0.5 * fit_quadratic(I, ser) / np.ptp(ser)


def qaoa_stats(E, n, params, p):
    En = qa.normalize(E)
    psi = qa.qaoa_state(params[:p], params[p:], En, n)
    prob = np.abs(psi) ** 2
    order = np.argsort(E)
    p_opt = float(prob[order[0]])
    top = max(1, int(0.01 * len(E)))
    p_top = float(prob[order[:top]].sum())
    mean = float(prob @ E)
    r = (E.max() - mean) / (E.max() - E.min())
    shots95 = float(np.log(0.05) / np.log(1 - p_opt)) if p_opt < 1 else 1.0
    return dict(p_opt=p_opt, p_top1=p_top, ratio=float(r), shots95=shots95)


def main(bits_list=(2, 3, 4, 5, 6), ps=(1, 2, 3, 4)):
    coef = surrogate_coef()
    res = []
    prev = {}
    for b in bits_list:
        n = 3 * b
        t0 = time.time()
        Q, c = surrogate_to_qubo(coef, (b, b, b), (2**b,) * 3)
        B = all_bitstrings(n)
        E = qubo_energy(Q, c, B)
        nz = int(np.count_nonzero(np.abs(np.triu(Q, 1)) > 1e-12))
        row = dict(n=n, bits=b, couplings=nz, cnot_layer=2 * nz, p_random=1.0 / 2**n, qaoa={})
        En = qa.normalize(E)
        cur = {}
        for p in ps:
            # V2 (amélioration) : deux initialisations, on garde la meilleure :
            #   - INTERP : angles optimaux de p - 1 (même n) interpolés à p (Zhou et al., 2020) ;
            #   - transfert : angles optimaux de la taille précédente (même p).
            # La première version (départs aléatoires) plafonnait dès p = 3.
            inits = []
            if p == 1 and n <= 9:
                r = qa.optimize_qaoa(En, n, p, n_restarts=8, seed=p)
            else:
                if p > 1:
                    inits.append(qa.interp_init(cur[p - 1], p))
                if p in prev:
                    inits.append(prev[p])
                best = None
                for x0 in inits:
                    rr = qa.optimize_qaoa(En, n, p, n_restarts=0, seed=p, init=x0)
                    if best is None or rr.fun < best.fun:
                        best = rr
                r = best
            cur[p] = r.x
        prev = cur
        for p in ps:
            r_x = cur[p]
            st = qaoa_stats(E, n, r_x, p)
            st.update(cnot_total=2 * nz * p)
            row["qaoa"][p] = st
        # recuit simulé : fraction de lectures qui trouvent l'optimum
        _, _, fin, _ = simulated_annealing(Q, c, n_sweeps=100, n_reads=50, seed=0, return_states=True)
        row["sa_success"] = float(np.mean(np.abs(fin - E.min()) < 1e-9 * max(1.0, abs(E.min()))))
        row["time_s"] = time.time() - t0
        res.append(row)
        print(f"n={n:2d} couplages={nz:3d} P(opt) QAOA p=1..4 = "
              + ", ".join(f"{row['qaoa'][p]['p_opt']:.4f}" for p in ps)
              + " ; 1 % meilleur " + ", ".join(f"{row['qaoa'][p]['p_top1']:.2f}" for p in ps)
              + f"  (hasard {1 / 2**n:.1e}), recuit {row['sa_success']:.2f}, {row['time_s']:.0f} s", flush=True)
    with open(OUT, "w") as f:
        json.dump(res, f, indent=1)
    return res


if __name__ == "__main__":
    main()
