"""Itération V3, lot L6 : les méthodes sur le problème mis à l'échelle (CST, 30 bits, ~10^9 profils).

Seconde campagne (probleme_cst.FILTRE = True) : la géométrie (faces sans croisement, épaisseur >= 8 %) est
contrôlée avant de simuler, pour toutes les méthodes, et une traînée > 200 counts compte comme un échec
d'XFOIL. La première campagne, sans filtre, est archivée dans resultats/ia_qubo_echelle_sansfiltre*.json :
le modèle en z y dépensait 59 % de son budget sur des géométries invalides.

Le front exact n'est plus connu. On mesure l'hypervolume des profils faisables simulés, normalisé par
celui du front de référence : le front non dominé de TOUS les profils faisables simulés par toutes les
méthodes et toutes les graines (lu dans le cache). Budget : 300 simulations, 8 graines.

Méthodes : hasard, NSGA-II (population 16), et la boucle avec chaque modèle :
    quadz_modele (l'approche actuelle : 66 coefficients en z, QUBO de 33 qubits résolu par recuit),
    fm_modele, fm_bn, bocs_modele,
    et un seul témoin non quantique (pas un QUBO) : hybride_modele (réseaux pour C_D et la SER, modèle affine
    pour C_L,max). mlp_modele est retiré : sur V2 il échoue à cause de son modèle de contrainte.

    python3 -m ia_qubo.experiences_echelle          -> resultats/ia_qubo_echelle.json
"""
import json
import os
import sys
import time

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from optim.nsga2 import nsga2  # noqa: E402
from optim.pareto import pareto_front, hypervolume_2d, Normalizer  # noqa: E402
from ia_qubo.boucle import boucle_ia  # noqa: E402
from ia_qubo import probleme_cst as P  # noqa: E402

HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUT = os.path.join(HERE, "resultats", "ia_qubo_echelle.json")
METHODS = ["hasard", "nsga2_16", "quadz_modele", "fm_modele", "fm_bn", "bocs_modele", "hybride_modele"]
BUDGET, N_SEEDS = 300, 8


def _run_one(args):
    method, seed = args
    pb = P.ProblemeSimule()
    t0 = time.time()
    if method == "hasard":
        rng = np.random.default_rng(seed)
        while pb.n_sims < BUDGET:
            pb(P.init_aleatoire(rng, 1)[0])
    elif method.startswith("nsga2"):
        nsga2(pb, pop_size=int(method.split("_")[1]), n_gen=200, seed=seed, n_levels=P.SPEC["levels"],
              max_sims=BUDGET)
    else:
        modele, contrainte = method.split("_")
        boucle_ia(pb, P.SPEC, modele=modele, contrainte="modele" if contrainte == "modele" else "boite_noire",
                  solveur="recuit", seed=seed, budget=BUDGET)
    return [list(i) for i in pb.seen[:BUDGET]], time.time() - t0


def post_traitement(runs):
    cache = P._charge_cache()
    # front de référence : les profils de CETTE campagne seulement (le cache peut en contenir d'autres)
    vus = {tuple(s) for lst in runs.values() for seen, _ in lst for s in seen}
    feas = {k: r for k, r in cache.items() if r["feasible"] and k in vus}
    keys = list(feas)
    F = np.array([[feas[k]["CD_counts"], feas[k]["SER_db"]] for k in keys])
    fr = pareto_front(F)
    norm = Normalizer(F)
    ref = np.array([1.1, 1.1])
    hv_ref = hypervolume_2d(norm(F[fr]), ref)
    front_ref = {keys[i] for i in fr}
    res = dict(n_simules_total=len(vus), filtre=P.FILTRE, cd_aberrant=P.CD_ABERRANT, n_faisables_total=len(feas), front_ref=[list(keys[i]) for i in fr],
               front_ref_F=F[fr].tolist(), hv_ref=hv_ref)
    grille = list(range(20, BUDGET + 1, 20))
    for m, lst in runs.items():
        traces, fracs, nfeas = [], [], []
        for seen, _ in lst:
            seen = [tuple(s) for s in seen]
            tr = []
            for k in grille:
                pts = [cache[s] for s in seen[:k] if cache[s]["feasible"]]
                if pts:
                    Fk = np.array([[r["CD_counts"], r["SER_db"]] for r in pts])
                    tr.append(hypervolume_2d(norm(Fk[pareto_front(Fk)]), ref) / hv_ref)
                else:
                    tr.append(0.0)
            traces.append(tr)
            fracs.append(len(front_ref & set(seen)) / len(front_ref))
            nfeas.append(float(np.mean([cache[s]["feasible"] for s in seen])))
        T = np.array(traces)
        res[m] = dict(grille=grille, hv_median=np.median(T, 0).tolist(), hv_q25=np.percentile(T, 25, 0).tolist(),
                      hv_q75=np.percentile(T, 75, 0).tolist(), hv_final=T[:, -1].tolist(),
                      part_front_ref=fracs, part_faisables=nfeas, secondes=[t for _, t in lst])
    return res


if __name__ == "__main__":
    from multiprocessing import Pool
    names = [m for m in METHODS if m in sys.argv] or METHODS
    jobs = [(m, s) for m in names for s in range(N_SEEDS)]
    jobs.sort(key=lambda j: 0 if j[0].startswith(("mlp", "hybride", "bocs")) else 1)
    workers = max(1, min(11, (os.cpu_count() or 2) - 1))
    t0 = time.time()
    with Pool(workers) as pool:
        out = pool.map(_run_one, jobs, chunksize=1)
    runs = {m: [o for j, o in zip(jobs, out) if j[0] == m] for m in names}
    with open(OUT.replace(".json", "_brut.json"), "w") as f:
        json.dump(runs, f)
    res = post_traitement(runs)
    res["duree_s"] = time.time() - t0
    with open(OUT, "w") as f:
        json.dump(res, f, default=lambda o: o.item() if hasattr(o, "item") else str(o))
    print(f"{res['n_simules_total']} profils simulés au total, {res['n_faisables_total']} faisables, "
          f"front de référence : {len(res['front_ref'])} profils")
    for m in names:
        r = res[m]
        print(f"   {m:13s} HV/HV_ref médian à 100, 200, 300 : "
              f"{r['hv_median'][4]:.3f} {r['hv_median'][9]:.3f} {r['hv_median'][-1]:.3f} ; "
              f"part du front de réf. {np.median(r['part_front_ref']):.2f} ; faisables {np.mean(r['part_faisables']):.2f} ; "
              f"{np.mean(r['secondes']) / 60:.1f} min par graine", flush=True)
    print(f"durée totale {res['duree_s'] / 60:.0f} min -> {OUT}")
