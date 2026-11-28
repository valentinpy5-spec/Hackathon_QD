"""Itération V3, lot L6 : boucle « modèle appris -> sélection -> simulation », générique.

Même structure que la boucle QUBO de la V2 (optim/qubo_loop.py) : 10 profils initiaux, 5 poids w,
un profil par poids et par itération, jamais deux fois le même. On choisit séparément :

    modele      "quadz" (approche actuelle), "fm" (machine de factorisation, FMQA),
                "bocs" (quadratique bayésien sur les bits, tirage de Thompson), "mlp" (réseau de neurones),
                "hybride" (réseaux de neurones pour C_D et la SER, modèle affine en z pour C_L,max)
    contrainte  "modele" : C_L,max modélisé (affine en z) et pénalisé avec variables d'écart, comme en V2
                "boite_noire" : aucun modèle de contrainte ; les profils infaisables entrent dans les
                données avec une valeur dégradée (pratique de FMQA), le modèle apprend seul à les éviter
    solveur     "exact" (énumération, jusqu'à 22 qubits) ou "recuit" (recuit simulé vectorisé)

Avec modele="quadz", contrainte="modele", solveur="exact", la boucle reproduit exactement qubo_loop
(vérifié par verifier_boucle()).
"""
from itertools import product

import numpy as np

from optim.qubo_loop import (fit_linear, grow, penalty_index_nonzero, penalty_linear_geq, lin_bits,
                             predict as predict_z, z_of)
from quantique.qubo import all_bitstrings, qubo_energy
from .modeles import QuadZ, FactorizationMachine, BayesQuadBits, MLPEnsemble, encode, decode_bits

WEIGHTS = (0.0, 0.25, 0.5, 0.75, 1.0)
PENALITE_INFAISABLE = 0.2      # mode boîte noire : un infaisable vaut max(faisables) + 0,2 x étendue


# ------------------------------------------------------------ recuit simulé vectorisé
def recuit(Q, c, n_sweeps=300, n_reads=64, T0=None, T1=None, seed=0):
    """Recuit simulé (Metropolis, un bit à la fois), toutes les lectures en parallèle (vectorisé numpy).
    Renvoie les états finaux (n_reads, n) et leurs énergies."""
    rng = np.random.default_rng(seed)
    n = Q.shape[0]
    S = np.triu(Q, 1)
    S = S + S.T
    d = np.diag(Q).copy()
    scale = np.abs(Q).max()
    T0 = T0 if T0 is not None else 2.0 * scale
    T1 = T1 if T1 is not None else 0.01 * scale
    B = rng.integers(0, 2, (n_reads, n)).astype(float)
    for T in np.geomspace(T0, T1, n_sweeps):
        for i in rng.permutation(n):
            dE = (1 - 2 * B[:, i]) * (d[i] + B @ S[i])
            acc = (dE <= 0) | (rng.random(n_reads) < np.exp(-np.clip(dE, 0, None) / T))
            B[acc, i] = 1 - B[acc, i]
    return B, qubo_energy(Q, c, B)


def classer(Q, c, solveur, rng, n_design, T_obj=None):
    """Chaînes classées par énergie croissante -> liste de chaînes de conception (n_design premiers bits)."""
    n = Q.shape[0]
    if solveur == "exact":
        Bs = all_bitstrings(n)
        E = qubo_energy(Q, c, Bs)
        order = np.argsort(E)[:600]
        states = Bs[order]
    else:
        # température réglée sur l'objectif (et non sur les pénalités, comme dans recuit.py) : sur les QUBO
        # de la boucle V2, la meilleure de 64 lectures trouve l'optimum dans 67 % des cas, contre 39 %
        # avec la température du code V2 (mesure : cours IA et QUBO, chapitre sur le passage à l'échelle)
        T0, T1 = (2.0 * T_obj, 0.01 * T_obj) if T_obj else (None, None)
        states, E = recuit(Q, c, n_sweeps=200, n_reads=128, T0=T0, T1=T1, seed=int(rng.integers(1 << 30)))
        states = states[np.argsort(E)]
    return states[:, :n_design]


# ------------------------------------------------------------ contraintes
def ajoute_contraintes(Q, c, coef_cl, spec, cl_min, P=3.0, n_slack=3, cl_weight=25.0, modele_cl=True):
    """Ajoute à un QUBO de conception (n_design bits) les mêmes pénalités qu'en V2 :
    t >= t_min (indice 0 interdit, Rosenberg, si spec['t_var'] n'est pas None) et, si modele_cl,
    C_L,max affine >= cl_min (variables d'écart)."""
    bits, levels = spec["bits"], spec["levels"]
    n = Q.shape[0]
    extra = (1 if spec.get("t_var") is not None else 0) + (n_slack if modele_cl else 0)
    Q = grow(Q, extra)
    k = n
    if spec.get("t_var") is not None:
        tv = spec["t_var"]
        first = sum(bits[:tv])
        Q, c = penalty_index_nonzero(Q, c, bits=(first, first + 1, first + 2), y=k, P=P)
        k += 1
    if modele_cl:
        const, lin = lin_bits(coef_cl, bits, levels)
        lmax = const + sum(max(a, 0) for _, a in lin)
        Q, c = penalty_linear_geq(Q, c, lin, const, cl_min, list(range(k, k + n_slack)),
                                  max(lmax - cl_min, 1e-3), P * cl_weight)
        k += n_slack
    return Q, c


# ------------------------------------------------------------ fabrique de modèles
def fabrique(modele, spec, seed):
    n = sum(spec["bits"])
    if modele == "quadz":
        return QuadZ(spec["bits"], spec["levels"])
    if modele == "fm":
        return FactorizationMachine(n, k=spec.get("fm_k", 4), seed=seed)
    if modele == "bocs":
        return BayesQuadBits(n, seed=seed)
    if modele in ("mlp", "hybride"):
        return MLPEnsemble(spec["bits"], spec["levels"], epochs=spec.get("mlp_epochs", 1500),
                           n_models=spec.get("mlp_models", 5), seed=seed)
    raise ValueError(modele)


def recherche_directe(score, spec, rng, exclus, n_alea=20000, n_depart=20):
    """Pour un modèle qui n'est pas un QUBO : énumération si la grille est petite, sinon tirage
    aléatoire puis recherche locale (on change un indice d'une variable à la fois)."""
    levels = spec["levels"]
    total = int(np.prod(levels))
    if total <= 50000:
        I = np.array(list(product(*[range(L) for L in levels])))
        s = score(I)
        return [tuple(int(v) for v in I[j]) for j in np.argsort(s)]
    I = np.column_stack([rng.integers(0, L, n_alea) for L in levels])
    s = score(I)
    best = [I[j].copy() for j in np.argsort(s)[:n_depart]]
    out = []
    for x in best:
        fx = score(x[None])[0]
        amelio = True
        while amelio:
            amelio = False
            voisins = []
            for i, L in enumerate(levels):
                for v in range(L):
                    if v != x[i]:
                        y = x.copy()
                        y[i] = v
                        voisins.append(y)
            V = np.array(voisins)
            sv = score(V)
            j = int(np.argmin(sv))
            if sv[j] < fx - 1e-12:
                x, fx, amelio = V[j], sv[j], True
        out.append((fx, tuple(int(v) for v in x)))
    out.sort()
    ranked = [t for _, t in out] + [tuple(int(v) for v in I[j]) for j in np.argsort(s)]
    return ranked


# ------------------------------------------------------------ la boucle
def boucle_ia(problem, spec, modele="fm", contrainte="modele", solveur="exact", n_init=10,
              weights=WEIGHTS, budget=250, seed=0, stop_when=None, cl_min=1.2):
    """spec : dict(bits, levels, keys, t_var (indice de la variable d'épaisseur ou None), ...)."""
    rng = np.random.default_rng(seed)
    bits, levels, keys = spec["bits"], spec["levels"], spec["keys"]
    nd = sum(bits)
    tv = spec.get("t_var")
    direct = modele in ("mlp", "hybride")     # modèle non quadratique : recherche directe, pas de QUBO
    if spec.get("init") is not None:
        init = spec["init"](rng, n_init)
    else:
        all_idx = list(product(*[range(L) for L in levels]))
        init = [all_idx[j] for j in rng.permutation(len(all_idx))[:n_init]]
    for idx in init:
        problem(idx)
    it = 0
    while problem.n_sims < budget:
        it += 1
        rows = [problem.table[i] for i in problem.seen]
        I = np.array([[r[k] for k in keys] for r in rows])
        B = encode(I, bits)
        ok = np.array([r["feasible"] for r in rows])
        val = np.array([r["valid"] for r in rows])
        cd = np.array([r["CD_counts"] for r in rows], float)
        ser = np.array([r["SER_db"] for r in rows], float)
        cl = np.array([r["CLmax"] for r in rows], float)
        new = []
        if ok.sum() < 4:
            if spec.get("init") is not None:
                cand = spec["init"](rng, 50)
            else:
                cand = [all_idx[j] for j in rng.permutation(len(all_idx))]
            new = [x for x in cand if x not in problem._set][:2]
        else:
            s_cd, s_ser = max(np.ptp(cd[ok]), 1e-6), max(np.ptp(ser[ok]), 1e-6)
            vl = val & np.isfinite(cl)
            coef_cl = fit_linear(I[vl], cl[vl], levels=levels) if vl.sum() >= 2 else None
            modele_cl = contrainte == "modele" and coef_cl is not None
            if contrainte == "modele":
                m_cd = fabrique(modele, spec, seed + 7919 * it).fit(B[ok], cd[ok])
                m_ser = fabrique(modele, spec, seed + 7919 * it + 1).fit(B[ok], ser[ok])
                if modele == "mlp":
                    m_cl = fabrique(modele, spec, seed + 7919 * it + 2).fit(B[vl], cl[vl]) if vl.sum() >= 4 else None
                elif modele == "hybride":
                    m_cl = None
                    cl_aff = coef_cl
            for w in weights:
                if contrainte == "modele":
                    if direct:
                        def score(Ic, w=w):
                            Bc = encode(Ic, bits)
                            f = w * m_cd.predict(Bc) / s_cd + (1 - w) * m_ser.predict(Bc) / s_ser
                            if m_cl is not None:
                                f = f + 75.0 * np.maximum(0.0, cl_min - m_cl.predict(Bc)) ** 2
                            elif modele == "hybride" and cl_aff is not None:
                                pred = predict_z(cl_aff, np.asarray(Ic), levels)
                                f = f + 75.0 * np.maximum(0.0, cl_min - pred) ** 2
                            if tv is not None:
                                f = f + 3.0 * (np.asarray(Ic)[:, tv] == 0)
                            return f
                    else:
                        if modele == "bocs":
                            Qa, ca = m_cd.sample_qubo()
                            Qb, cb = m_ser.sample_qubo()
                        else:
                            Qa, ca = m_cd.to_qubo()
                            Qb, cb = m_ser.to_qubo()
                        Q = w * Qa / s_cd + (1 - w) * Qb / s_ser
                        c = w * ca / s_cd + (1 - w) * cb / s_ser
                else:
                    # boîte noire : l'objectif pondéré, et une valeur dégradée pour les infaisables
                    f = np.full(len(rows), np.nan)
                    f[ok] = w * cd[ok] / s_cd + (1 - w) * ser[ok] / s_ser
                    worst = np.nanmax(f) + PENALITE_INFAISABLE * max(np.nanmax(f) - np.nanmin(f), 1e-6)
                    f[~ok] = worst
                    m = fabrique(modele, spec, seed + 7919 * it + int(100 * w)).fit(B, f)
                    if direct:
                        def score(Ic, m=m):
                            s = m.predict(encode(Ic, bits))
                            if tv is not None:
                                s = s + 3.0 * (np.asarray(Ic)[:, tv] == 0)
                            return s
                    else:
                        Q, c = m.sample_qubo() if modele == "bocs" else m.to_qubo()
                if direct:
                    ranked = recherche_directe(score, spec, rng, problem._set)
                else:
                    T_obj = float(np.abs(Q).max())
                    Q, c = ajoute_contraintes(Q, c, coef_cl, spec, cl_min, modele_cl=(contrainte == "modele" and modele_cl))
                    states = classer(Q, c, solveur, rng, nd, T_obj=T_obj)
                    ranked = [tuple(int(v) for v in x) for x in decode_bits(states, bits)]
                filtre = spec.get("filtre")
                for idx in ranked:
                    if filtre is not None and not filtre(idx):
                        continue          # post-traitement classique des échantillons : géométrie gratuite
                    if idx not in problem._set and idx not in new:
                        new.append(idx)
                        break
            if not new:
                cand = spec["init"](rng, 50) if spec.get("init") is not None else \
                    [all_idx[j] for j in rng.permutation(len(all_idx))]
                new = [x for x in cand if x not in problem._set][:1]
        for idx in new:
            if problem.n_sims >= budget:
                break
            problem(idx)
        if stop_when and stop_when(problem):
            break
    return dict(iterations=it)


def verifier_boucle():
    """La boucle générique (quadz, modèle, exact) doit simuler exactement les mêmes profils que qubo_loop."""
    from optim.problem import CountedProblem
    from optim.problem_v2 import build_table_v2, with_objective, SpaceV2, KEYS
    from optim.qubo_loop import qubo_loop
    from optim.experiences_v2 import CONS
    SP = SpaceV2()
    T = with_objective(build_table_v2(), "S2", "MoM")
    spec = dict(bits=SP.bits, levels=SP.n_levels, keys=KEYS, t_var=2)
    ok = True
    for seed in (0, 1, 2):
        a, b = CountedProblem(T, keys=KEYS), CountedProblem(T, keys=KEYS)
        qubo_loop(a, n_init=10, solver="exact", seed=seed, budget=60, cons=CONS, bits=SP.bits,
                  levels=SP.n_levels, keys=KEYS, lin_col="CLmax")
        boucle_ia(b, spec, modele="quadz", contrainte="modele", solveur="exact", seed=seed, budget=60)
        ok &= a.seen == b.seen
        print(f"graine {seed} : mêmes profils, dans le même ordre : {a.seen == b.seen}")
    return ok
